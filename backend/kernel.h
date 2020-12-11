#ifndef WEFT_BACKEND_KERNEL_H
#define WEFT_BACKEND_KERNEL_H

#include <google/protobuf/repeated_field.h>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "weft.grpc.pb.h"

namespace weft::kernel {

class Module {
 public:
  Module(uint64_t handle, std::string ptx)
      : handle_{handle}, ptx_{std::move(ptx)} {}

  constexpr uint64_t handle() const noexcept { return handle_; }
  std::string ptx() const noexcept { return ptx_; }

 private:
  uint64_t handle_;
  std::string ptx_;
};

struct Param {
  size_t size;
  bool is_pointer;
  bool is_const;

  Param(size_t size, bool is_pointer, bool is_const)
      : size{size}, is_pointer{is_pointer}, is_const{is_const} {}
};

class Function {
 public:
  Function(uint64_t handle, const Module &module, std::string name,
           std::vector<Param> params)
      : handle_{handle},
        module_{module},
        name_{std::move(name)},
        params_{std::move(params)} {}

  void Launch(
      uint32_t gridDimX, uint32_t gridDimY, uint32_t gridDimZ,
      uint32_t blockDimX, uint32_t blockDimY, uint32_t blockDimZ,
      uint32_t sharedMemBytes,
      const google::protobuf::RepeatedPtrField<weft::FunctionMetadata_Param>
          &params) const;

  constexpr uint64_t handle() const noexcept { return handle_; }
  const Module &module() const noexcept { return module_; }
  std::string name() const noexcept { return name_; }
  const std::vector<Param> &params() const noexcept { return params_; }

 private:
  uint64_t handle_;
  const Module &module_;
  std::string name_;
  std::vector<Param> params_;
};

uint64_t add_ptx(std::string ptx);

const Function &get_function(uint64_t f_handle);
const Function &add_function(uint64_t m_handle, std::string name,
                             std::vector<Param> params);

}  // namespace weft::kernel

#endif  // WEFT_BACKEND_KERNEL_H
