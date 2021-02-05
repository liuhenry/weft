#include "device.h"

#include <cuda.h>

#include <iostream>

#include "CUDA_samples/helper_cuda_drvapi.h"
#include "weft.grpc.pb.h"

using google::protobuf::RepeatedPtrField;
using weft::FunctionMetadata_Param;
namespace weft {

inline int get_max_concurrency_from_version(int major, int minor) {
  // Refer to ConvertSMVer2Cores
  typedef struct {
    int CC;  // 0xMm (hexidecimal notation), M = CC Major version,
    // and m = CC minor version
    int MaxConcurrency;
  } sCCtoConcurrency;

  sCCtoConcurrency maxGpuArchKernelConcurrency[] = {
      {0x35, 32},  {0x37, 32},  {0x50, 32},  {0x52, 32},  {0x53, 16},
      {0x60, 128}, {0x61, 32},  {0x62, 16},  {0x70, 128}, {0x72, 16},
      {0x75, 128}, {0x80, 128}, {0x86, 128}, {-1, -1}};

  int index = 0;

  while (maxGpuArchKernelConcurrency[index].CC != -1) {
    if (maxGpuArchKernelConcurrency[index].CC == ((major << 4) + minor)) {
      return maxGpuArchKernelConcurrency[index].MaxConcurrency;
    }

    index++;
  }
}

Device::Device(int device_idx) : device_idx_{device_idx} {
  checkCudaErrors(cuDeviceGet(&device_, device_idx_));
  checkCudaErrors(
      cuDeviceGetName(device_name_.data(), device_name_.size(), device_));

  checkCudaErrors(cuDeviceGetAttribute(
      &compute_capability_major_, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
      device_));
  checkCudaErrors(cuDeviceGetAttribute(
      &compute_capability_minor_, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
      device_));

  max_concurrent_kernels_ = get_max_concurrency_from_version(
      compute_capability_major_, compute_capability_minor_);

  std::clog << "Device " << device_idx_ << ": \"" << device_name_
            << "\" (Compute " << compute_capability_major_ << "."
            << compute_capability_minor_ << " — "
            << "Max Kernel Concurrency: " << max_concurrent_kernels_ << ")\n";

  checkCudaErrors(cuDevicePrimaryCtxRetain(&context_, device_));
  checkCudaErrors(cuCtxSetCurrent(context_));

  for (int i = 0; i < max_concurrent_kernels_; i++) {
    CUstream stream;
    checkCudaErrors(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    stream_pool.bounded_push(std::move(stream));
  }
}

Device::~Device() { checkCudaErrors(cuDevicePrimaryCtxRelease(device_)); }

}  // namespace weft
