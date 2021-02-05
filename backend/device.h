#ifndef WEFT_BACKEND_DEVICE_H
#define WEFT_BACKEND_DEVICE_H

#include <cuda.h>

#include <boost/lockfree/policies.hpp>
#include <boost/lockfree/queue.hpp>
#include <string>

namespace weft {

// Forward declarations
class Device {
 public:
  explicit Device(int device_idx);
  ~Device();

  Device(const Device &) = delete;
  Device &operator=(const Device &) = delete;

  Device(Device &&) = default;
  Device &operator=(Device &&) = default;

  operator CUdevice() const { return device_; }
  operator CUcontext() const { return context_; }
  boost::lockfree::queue<CUstream, boost::lockfree::capacity<128>> stream_pool;

  friend std::ostream &operator<<(std::ostream &os, const Device &device) {
    os << device.device_idx_;
    return os;
  }

 private:
  int device_idx_;
  CUdevice device_;
  CUcontext context_;

  // Device properties
  std::string device_name_ = std::string(64, 0);
  int compute_capability_major_;
  int compute_capability_minor_;
  int max_concurrent_kernels_;
};

}  // namespace weft

#endif  // WEFT_BACKEND_DEVICE_H
