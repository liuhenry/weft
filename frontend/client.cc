#include "client.h"

#include <grpc/grpc.h>
#include <grpcpp/client_context.h>

#include <boost/range/adaptor/indexed.hpp>
#include <iostream>
#include <memory>
#include <string_view>
#include <vector>

#include "nvrtc/kernel_parser.h"
#include "weft.grpc.pb.h"

namespace weft {

using grpc::ClientContext;
using grpc::ClientReader;
using grpc::ClientWriter;
using grpc::Status;

constexpr size_t chunk_size = 64 * 1024;

uint64_t CudaDriverClient::MemAlloc(size_t size) {
  ClientContext context;
  Size request;
  DevicePointer response;

  request.set_size(size);
  Status status = stub_->MemAlloc(&context, request, &response);
  if (!status.ok()) {
    std::cerr << "RPC MemAlloc Failed!\n\t" << status.error_message() << "\n";
  }
  return response.handle();
}

void CudaDriverClient::MemFree(uint64_t dptr) {
  ClientContext context;
  DevicePointer request;
  Empty response;

  request.set_handle(dptr);
  Status status = stub_->MemFree(&context, request, &response);
  if (!status.ok()) {
    std::cerr << "RPC MemFree Failed!\n\t" << status.error_message() << "\n";
  }
}

void CudaDriverClient::MemcpyHtoD(uint64_t dptr, std::string_view src) {
  ClientContext context;
  MemoryWrite chunk;
  Empty response;

  std::unique_ptr<ClientWriter<MemoryWrite>> writer(
      stub_->MemcpyHtoD(&context, &response));

  // Write metadata
  chunk.mutable_dptr()->set_handle(dptr);
  writer->Write(chunk);

  // Write data
  for (unsigned i = 0; i < src.length(); i += chunk_size) {
    auto substr = src.substr(i, chunk_size);
    chunk.mutable_chunk()->set_data(substr.data(), substr.length());
    writer->Write(chunk);
  }

  // Finish
  writer->WritesDone();
  Status status = writer->Finish();
  if (!status.ok()) {
    std::cerr << "RPC MemcpyHtoD Failed!\n\t" << status.error_message() << "\n";
  }
}

void CudaDriverClient::MemcpyDtoH(void *dst, uint64_t sptr, size_t size) {
  ClientContext context;
  MemoryRead request;
  MemoryChunk chunk;

  request.mutable_dptr()->set_handle(sptr);
  request.mutable_size()->set_size(size);

  std::unique_ptr<ClientReader<MemoryChunk>> reader(
      stub_->MemcpyDtoH(&context, request));

  auto dstChar = static_cast<char *>(dst);
  while (reader->Read(&chunk)) {
    memcpy(dstChar, chunk.data().data(), chunk.data().length());
    dstChar += chunk.data().length();
  }
  Status status = reader->Finish();
  if (!status.ok()) {
    std::cerr << "RPC MemcpyDtoH Failed!\n\t" << status.error_message() << "\n";
  }
}

uint64_t CudaDriverClient::ModuleGetFunction(
    uint64_t hmod, std::string name,
    const std::vector<weft::nvrtc::Param> &params) {
  ClientContext context;
  FunctionMetadata request;
  Function response;

  request.mutable_module()->set_handle(hmod);
  request.set_function_name(name);

  for (const auto &param : params) {
    auto request_param = request.add_params();
    request_param->set_size(param.size());
    request_param->set_is_pointer(param.is_pointer());
    request_param->set_is_const(param.is_const());
  }

  Status status = stub_->ModuleGetFunction(&context, request, &response);
  if (!status.ok()) {
    std::cerr << "RPC ModuleGetFunction Failed!\n\t" << status.error_message()
              << "\n";
  }
  return response.handle();
}

uint64_t CudaDriverClient::ModuleLoadData(std::string image) {
  ClientContext context;
  PTX request;
  Module response;

  request.set_str(image);
  Status status = stub_->ModuleLoadData(&context, request, &response);
  if (!status.ok()) {
    std::cerr << "RPC ModuleLoadData Failed!\n\t" << status.error_message()
              << "\n";
  }
  return response.handle();
}

void CudaDriverClient::LaunchKernel(
    uint64_t f, uint32_t gridDimX, uint32_t gridDimY, uint32_t gridDimZ,
    uint32_t blockDimX, uint32_t blockDimY, uint32_t blockDimZ,
    uint32_t sharedMemBytes, uint64_t hStream,
    const std::vector<weft::nvrtc::Param> &metadata, void *kernelParams[]) {
  ClientContext context;
  KernelLaunch request;
  Empty response;

  request.set_f(f);
  request.set_griddimx(gridDimX);
  request.set_griddimy(gridDimY);
  request.set_griddimz(gridDimZ);
  request.set_blockdimx(blockDimX);
  request.set_blockdimy(blockDimY);
  request.set_blockdimz(blockDimZ);
  request.set_sharedmembytes(sharedMemBytes);
  request.set_hstream(hStream);

  auto kernelParams_char = reinterpret_cast<char **>(kernelParams);
  for (const auto &param : metadata | boost::adaptors::indexed(0)) {
    auto request_param = request.add_params();
    request_param->set_size(param.value().size());
    request_param->set_pointee_size(param.value().pointee_size());
    request_param->set_is_pointer(param.value().is_pointer());
    request_param->set_is_const(param.value().is_const());
    request_param->set_data(kernelParams[param.index()], param.value().size());
  }

  Status status = stub_->LaunchKernel(&context, request, &response);
  if (!status.ok()) {
    std::cerr << "RPC LaunchKernel Failed!\n\t" << status.error_message()
              << "\n";
  }
}

}  // namespace weft
