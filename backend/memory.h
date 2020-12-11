#ifndef WEFT_BACKEND_MEMORY_H
#define WEFT_BACKEND_MEMORY_H

#include <cuda.h>

#include <memory>
#include <unordered_map>

namespace weft::memory {

class Block {
 public:
  Block(uint64_t handle, size_t size)
      : handle_{handle},
        size_{size},
        data_{std::make_unique<unsigned char[]>(size)} {}

  // TODO: write destructor for device_ptrs

  constexpr uint64_t handle() const noexcept { return handle_; }
  constexpr size_t size() const noexcept { return size_; }
  void* data() const noexcept { return data_.get(); }

  CUdeviceptr* device_ptr(CUdevice device);

 private:
  uint64_t handle_;
  size_t size_;
  std::unique_ptr<unsigned char[]> data_;
  std::unordered_map<CUdevice, CUdeviceptr> device_ptrs_;
};

uint64_t malloc(size_t size);
void free(uint64_t handle);
Block& get_block(uint64_t handle);

}  // namespace weft::memory

#endif  // WEFT_BACKEND_MEMORY_H
