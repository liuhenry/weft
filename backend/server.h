#ifndef WEFT_BACKEND_SERVER_H
#define WEFT_BACKEND_SERVER_H

#include <grpc/grpc.h>
#include <grpcpp/server.h>
#include <grpcpp/server_context.h>

#include "device.h"
#include "weft.grpc.pb.h"

namespace weft {

class CudaDriverImpl final : public CudaDriver::Service {
 public:
  CudaDriverImpl();

  grpc::Status MemAlloc(grpc::ServerContext* context, const Size* request,
                        DevicePointer* response) override;
  grpc::Status MemFree(grpc::ServerContext* context,
                       const DevicePointer* request,
                       Empty* /*response*/) override;
  grpc::Status MemcpyHtoD(grpc::ServerContext* context,
                          grpc::ServerReader<MemoryWrite>* request,
                          Empty* /*response*/) override;
  grpc::Status MemcpyDtoH(grpc::ServerContext* context,
                          const MemoryRead* request,
                          grpc::ServerWriter<MemoryChunk>* response) override;

  grpc::Status ModuleGetFunction(grpc::ServerContext* context,
                                 const FunctionMetadata* request,
                                 Function* response) override;
  grpc::Status ModuleLoadData(grpc::ServerContext* context, const PTX* request,
                              Module* response) override;

  grpc::Status LaunchKernel(grpc::ServerContext* context,
                            const KernelLaunch* request,
                            Empty* /*response*/) override;

 private:
  int device_count_;
  std::unique_ptr<Device[]> devices_;

  int CuInitialize();
};

}  // namespace weft

#endif  // WEFT_BACKEND_SERVER_H
