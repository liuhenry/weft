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
#include "weft.grpc.pb.h"

namespace weft {

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerReader;
using grpc::ServerWriter;
using grpc::Status;

constexpr size_t chunk_size = 64 * 1024;

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
  auto func = kernel::get_function(request->f());
  const kernel::ExecutionArgs execution{*request};
  scheduler_.schedule(func, execution);

  return Status::OK;
}

}  // namespace weft
