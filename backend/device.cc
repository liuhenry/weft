#include "device.h"

#include <cuda.h>

#include <iostream>

#include "CUDA_samples/helper_cuda_drvapi.h"

namespace weft {

Device::Device(int device_idx)
    : initialized_{true}, device_idx_{device_idx}, context_{} {
  checkCudaErrors(cuDeviceGet(&device_, device_idx_));

  checkCudaErrors(
      cuDeviceGetName(&device_name_[0], device_name_.size(), device_));

  int major = 0, minor = 0;
  checkCudaErrors(cuDeviceComputeCapability(&major, &minor, device_));
  checkCudaErrors(cuDeviceGetProperties(&device_props_, device_));
  std::clog << "Device " << device_idx_ << ": \"" << device_name_
            << "\" (Compute " << major << "." << minor << ")\n";
  std::clog << "\tsharedMemPerBlock: " << device_props_.sharedMemPerBlock
            << "\n";
  std::clog << "\tconstantMemory   : " << device_props_.totalConstantMemory
            << "\n";
  std::clog << "\tregsPerBlock     : " << device_props_.regsPerBlock << "\n";
  std::clog << "\tclockRate        : " << device_props_.clockRate << "\n";
  std::clog << "\n";
}

Context::Context(Device *device) : initialized_(true), device_(device) {
  checkCudaErrors(cuCtxCreate(&context_, CU_CTX_SCHED_AUTO, *device_));
  checkCudaErrors(cuCtxPopCurrent(nullptr));

  thread_ = std::thread(&Context::Loop, this);
  std::clog << "<CUDA Device=" << *device_ << ", Context=" << *this
            << ", Thread=" << thread_.get_id()
            << "> - Context::Loop() Launched...\n";
}

Context::~Context() {
  if (thread_.joinable()) thread_.join();
  if (initialized_) cuCtxDestroy(context_);
}

void Context::Loop() {
  checkCudaErrors(cuCtxPushCurrent(context_));
  while (true) {
  }
}

}  // namespace weft
