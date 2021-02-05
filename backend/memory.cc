#include "memory.h"

#include <cuda.h>

#include <cstring>
#include <functional>
#include <iostream>
#include <mutex>
#include <random>
#include <unordered_map>

#include "CUDA_samples/helper_cuda_drvapi.h"
namespace weft::memory {

static std::unordered_map<uint64_t, Block> mmap;

auto rand = std::bind(std::uniform_int_distribution<uint64_t>{},
                      std::mt19937(std::random_device{}()));

CUdeviceptr* Block::device_ptr(CUdevice device) {
  auto* ptr = &device_ptrs_[device];
  if (!*ptr) cuMemAlloc(ptr, size_);
  return ptr;
}

void Block::write_back(const CUdevice& device, const CUstream& stream) {
  auto* d_ptr = &device_ptrs_[device];

  // Thread-safe copy original
  std::call_once(orig_data_init_, [&]() {
    orig_data_ = std::make_unique<unsigned char[]>(size_);
    std::memcpy(orig_data_.get(), data_.get(), size_);
  });

  // Copy to temp buffer
  auto buf = std::make_unique<unsigned char[]>(size_);
  checkCudaErrors(cuMemcpyDtoHAsync(buf.get(), *d_ptr, size_, stream));

  // Check consistency and perform real copy
  for (size_t i = 0; i < size_; ++i) {
    if (orig_data_.get()[i] != buf[i]) {
      data_.get()[i] = buf[i];
    }
  }
}

uint64_t malloc(size_t size) {
  auto handle = rand();
  while (mmap.find(handle) != mmap.end()) {
    handle = rand();
  }

  mmap.emplace(std::piecewise_construct, std::forward_as_tuple(handle),
               std::forward_as_tuple(handle, size));
  return handle;
}

Block& get_block(uint64_t handle) { return mmap.at(handle); }

void free(uint64_t handle) { mmap.erase(handle); }

}  // namespace weft::memory
