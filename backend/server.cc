#include "server.h"

#include <cuda.h>
#include <grpc/grpc.h>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>

#include <string_view>

#include "CUDA_samples/drvapi_error_string.h"
#include "device.h"
#include "kernel.h"
#include "memory.h"

namespace weft {

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerReader;
using grpc::ServerWriter;
using grpc::Status;

constexpr size_t chunk_size = 64 * 1024;

CudaDriverImpl::CudaDriverImpl()
    : device_count_{CuInitialize()}, devices_{new Device[device_count_]} {
  for (int i = 0; i < device_count_; i++) {
    devices_[i] = Device{i};
  }
}

Status CudaDriverImpl::MemAlloc(ServerContext* context, const Size* request,
                                DevicePointer* response) {
  auto handle = memory::malloc(request->size());
  std::clog << "> VMM: MemAlloc " << request->size() << " bytes at " << handle
            << "\n";
  response->set_handle(handle);
  return Status::OK;
}

Status CudaDriverImpl::MemFree(ServerContext* context,
                               const DevicePointer* request,
                               Empty* /*response*/) {
  auto handle = request->handle();
  memory::free(handle);
  std::clog << "> VMM: MemFree " << handle << "\n";
  return Status::OK;
}

Status CudaDriverImpl::MemcpyHtoD(ServerContext* context,
                                  ServerReader<MemoryWrite>* request,
                                  Empty* /*response*/) {
  MemoryWrite chunk;

  // Initial read + get block
  request->Read(&chunk);
  const auto& block = memory::get_block(chunk.dptr().handle());

  // Write chunks to block
  auto block_data_ptr = static_cast<char*>(block.data());
  while (request->Read(&chunk)) {
    auto data = chunk.chunk().data();
    std::memcpy(block_data_ptr, data.data(), data.length());
    block_data_ptr += data.length();
  }

  std::clog << "> VMM: MemcpyHtoD " << block.handle() << "\n";
  return Status::OK;
}

Status CudaDriverImpl::MemcpyDtoH(ServerContext* context,
                                  const MemoryRead* request,
                                  ServerWriter<MemoryChunk>* response) {
  const auto& block = memory::get_block(request->dptr().handle());

  MemoryChunk chunk;
  std::string_view chunker(reinterpret_cast<char*>(block.data()),
                           request->size().size());  // TODO: unsigned?
  std::string_view substr;
  for (unsigned i = 0; i < chunker.length(); i += chunk_size) {
    substr = chunker.substr(i, chunk_size);
    chunk.set_data(substr.data(), substr.length());
    response->Write(chunk);
  }
  std::clog << "> VMM: MemcpyDtoH " << block.handle() << "\n";
  return Status::OK;
}

Status CudaDriverImpl::ModuleGetFunction(ServerContext* context,
                                         const FunctionMetadata* request,
                                         Function* response) {
  // Create server-side param data
  std::vector<kernel::Param> params;
  params.reserve(request->params_size());
  for (const auto& param : request->params()) {
    params.emplace_back(param.size(), param.is_pointer(), param.is_const());
  }

  // FIXME: move semantics for params
  auto function = kernel::add_function(request->module().handle(),
                                       request->function_name(), params);
  std::clog << "> Kernel: ModuleGetFunction " << function.handle() << "\n";
  response->set_handle(function.handle());
  return Status::OK;
}

Status CudaDriverImpl::ModuleLoadData(ServerContext* context,
                                      const PTX* request, Module* response) {
  auto handle = kernel::add_ptx(request->str());
  std::clog << "> Kernel: ModuleLoadData " << handle << "\n";
  response->set_handle(handle);
  return Status::OK;
}

Status CudaDriverImpl::LaunchKernel(ServerContext* context,
                                    const KernelLaunch* request,
                                    Empty* /*empty*/) {
  auto f = request->f();
  auto gridDimX = request->griddimx();
  auto gridDimY = request->griddimy();
  auto gridDimZ = request->griddimz();
  auto blockDimX = request->blockdimx();
  auto blockDimY = request->blockdimy();
  auto blockDimZ = request->blockdimz();
  auto sharedMemBytes = request->sharedmembytes();
  auto hStream = request->hstream();

  kernel::get_function(f).Launch(gridDimX, gridDimY, gridDimZ, blockDimX,
                                 blockDimY, blockDimZ, sharedMemBytes,
                                 request->params());

  return Status::OK;
}

int CudaDriverImpl::CuInitialize() {
  CUresult error_id = cuInit(0);

  if (error_id != CUDA_SUCCESS) {
    std::cerr << "cuInit(0) returned " << error_id << "\n-> "
              << getCudaDrvErrorString(error_id) << "\n";
    std::cerr << "Result = FAIL\n";
    exit(EXIT_FAILURE);
  }

  int device_count;
  error_id = cuDeviceGetCount(&device_count);

  if (error_id != CUDA_SUCCESS) {
    std::cerr << "cuDeviceGetCount returned " << error_id << "\n-> "
              << getCudaDrvErrorString(error_id) << "\n";
    std::cerr << "Result = FAIL\n";
    exit(EXIT_FAILURE);
  }

  if (device_count == 0) {
    std::clog << "There are no available device(s) that support CUDA\n";
  } else {
    std::clog << "Detected " << device_count << " CUDA Capable device(s)\n";
  }

  return device_count;
}

}  // namespace weft
