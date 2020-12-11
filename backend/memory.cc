#include "memory.h"

#include <cuda.h>

#include <cstring>
#include <functional>
#include <iostream>
#include <random>
#include <unordered_map>

namespace weft::memory {

static std::unordered_map<uint64_t, Block> mmap;

auto rand = std::bind(std::uniform_int_distribution<uint64_t>{},
                      std::mt19937(std::random_device{}()));

CUdeviceptr* Block::device_ptr(CUdevice device) {
  auto* ptr = &device_ptrs_[device];
  if (!*ptr) cuMemAlloc(ptr, size_);
  return ptr;
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
