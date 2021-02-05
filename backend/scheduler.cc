#include "scheduler.h"

#include <cuda.h>

#include <iostream>
#include <thread>
#include <vector>

#include "CUDA_samples/helper_cuda_drvapi.h"

namespace weft {

Scheduler::Scheduler() : device_count_{CuInitialize()} {
  devices_.reserve(device_count_);
  for (int i = 0; i < device_count_; i++) {
    devices_.emplace_back(i);
  }
}

int Scheduler::CuInitialize() {
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

void Scheduler::schedule(const kernel::Function &func,
                         const kernel::ExecutionArgs &execution) {
  std::vector<std::thread> threads;
  threads.reserve(device_count_);

  for (int i = 0; i < device_count_; i++) {
    auto execution_slice = execution;
    execution_slice.gridDimX = execution.gridDimX / device_count_;
    execution_slice.blockOffset = execution.gridDimX / device_count_ * i;
    threads.emplace_back(&kernel::Function::execute, func,
                         std::ref(devices_[i]), execution_slice);
  }

  for (auto &t : threads) {
    t.join();
  }
}

}  // namespace weft
