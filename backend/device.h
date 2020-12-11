#ifndef WEFT_BACKEND_DEVICE_H
#define WEFT_BACKEND_DEVICE_H

#include <cuda.h>

#include <memory>
#include <string>
#include <thread>

namespace weft {

// Forward declarations
class Device;
class Context;
class Stream;

class Device {
 public:
  Device() : initialized_(false){};
  Device(int device_idx);

  operator CUdevice() const { return device_; }

 private:
  bool initialized_;

  int device_idx_;
  CUdevice device_;
  std::string device_name_ = std::string(64, 0);
  CUdevprop device_props_;

  std::unique_ptr<Context> context_;
};

class Context {
 public:
  Context() : initialized_(false) {}
  Context(Device *device);

  Context(Context &&) = default;
  Context &operator=(Context &&) = default;

  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;

  ~Context();

  operator CUcontext() const { return context_; }

 private:
  bool initialized_;

  CUcontext context_;
  std::unique_ptr<Device> device_;

  std::thread thread_;

  void Loop();
};

class Stream {
 public:
 private:
};

}  // namespace weft

#endif  // WEFT_BACKEND_DEVICE_H
