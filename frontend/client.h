#ifndef WEFT_FRONTEND_CLIENT_H
#define WEFT_FRONTEND_CLIENT_H

#include <cuda.h>
#include <grpc/grpc.h>
#include <grpcpp/channel.h>

#include <memory>
#include <string_view>
#include <vector>

#include "nvrtc/kernel_parser.h"
#include "weft.grpc.pb.h"

namespace weft {

class CudaDriverClient {
 public:
  CudaDriverClient() {}
  CudaDriverClient(std::shared_ptr<grpc::Channel> channel)
      : stub_{CudaDriver::NewStub(channel)} {}

  uint64_t MemAlloc(size_t size);
  void MemFree(uint64_t dptr);
  void MemcpyHtoD(uint64_t dptr, std::string_view src);
  void MemcpyDtoH(void *dst, uint64_t sptr, size_t size);

  uint64_t ModuleGetFunction(uint64_t hmod, std::string name,
                             const std::vector<weft::nvrtc::Param> &params);
  uint64_t ModuleLoadData(std::string image);

  void LaunchKernel(uint64_t f, uint32_t gridDimX, uint32_t gridDimY,
                    uint32_t gridDimZ, uint32_t blockDimX, uint32_t blockDimY,
                    uint32_t blockDimZ, uint32_t sharedMemBytes,
                    uint64_t hStream,
                    const std::vector<weft::nvrtc::Param> &metadata,
                    void *kernelParams[]);

 private:
  std::unique_ptr<CudaDriver::Stub> stub_;
};

}  // namespace weft

#endif  // WEFT_FRONTEND_CLIENT_H
