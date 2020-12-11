#include "kernel.h"

#include <cuda.h>
#include <google/protobuf/repeated_field.h>

#include <boost/range/adaptor/indexed.hpp>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <unordered_map>

#include "CUDA_samples/helper_cuda_drvapi.h"
#include "memory.h"
#include "weft.grpc.pb.h"

using google::protobuf::RepeatedPtrField;
using weft::FunctionMetadata_Param;

namespace weft::kernel {

static std::unordered_map<uint64_t, Module> modules;
static std::unordered_map<uint64_t, Function> functions;

auto rand = std::bind(std::uniform_int_distribution<uint64_t>{},
                      std::mt19937(std::random_device{}()));

uint64_t add_ptx(std::string ptx) {
  auto handle = rand();
  while (modules.find(handle) != modules.end()) {
    handle = rand();
  }

  modules.emplace(std::piecewise_construct, std::forward_as_tuple(handle),
                  std::forward_as_tuple(handle, ptx));
  return handle;
}

const Function& get_function(uint64_t f_handle) {
  return functions.at(f_handle);
}

const Function& add_function(uint64_t m_handle, std::string name,
                             std::vector<Param> params) {
  auto const& module = modules.at(m_handle);
  auto f_handle = rand();
  while (functions.find(f_handle) != functions.end()) {
    f_handle = rand();
  }

  functions.emplace(std::piecewise_construct, std::forward_as_tuple(f_handle),
                    std::forward_as_tuple(f_handle, module, name, params));

  return functions.at(f_handle);
}

void Function::Launch(
    uint32_t gridDimX, uint32_t gridDimY, uint32_t gridDimZ, uint32_t blockDimX,
    uint32_t blockDimY, uint32_t blockDimZ, uint32_t sharedMemBytes,
    const RepeatedPtrField<FunctionMetadata_Param>& params) const {
  std::clog << "> LaunchKernel: " << handle_ << " (" << name_ << ")\n"
            << "\t Grid — X: " << gridDimX << ", Y: " << gridDimY
            << ", Z: " << gridDimZ << "\n"
            << "\t Block — X: " << blockDimX << ", Y: " << blockDimY
            << ", Z: " << blockDimZ << "\n"
            << "\t Shared Memory: " << sharedMemBytes << "\n";

  CUdevice device;
  CUmodule module;
  CUcontext cu_context;
  CUfunction kernel_addr;

  checkCudaErrors(cuDeviceGet(&device, 1));  // FIXME: schedule onto second GPU
  checkCudaErrors(cuCtxCreate(&cu_context, 0, device));

  checkCudaErrors(cuModuleLoadDataEx(&module, module_.ptx().c_str(), 0, 0, 0));
  checkCudaErrors(cuModuleGetFunction(&kernel_addr, module, name_.c_str()));

  std::vector<void*> args;
  for (const auto& param : params | boost::adaptors::indexed(0)) {
    void* ptr;
    if (param.value().is_pointer()) {
      auto& block = memory::get_block(*reinterpret_cast<uint64_t*>(
          const_cast<char*>(param.value().data().c_str())));
      auto d_ptr = block.device_ptr(device);  // Also cuMalloc's if we haven't
      if (param.value().is_const()) {
        checkCudaErrors(cuMemcpyHtoD(*d_ptr, block.data(), block.size()));
      }
      ptr = reinterpret_cast<void*>(d_ptr);
    } else {
      ptr = reinterpret_cast<void*>(
          const_cast<char*>(param.value().data().c_str()));
    }
    args.emplace_back(ptr);

    std::clog << "\t " << param.index() << " - Size: " << param.value().size()
              << ", Pointer: " << param.value().is_pointer()
              << ", Const: " << param.value().is_const() << "\n";
  }

  checkCudaErrors(cuLaunchKernel(kernel_addr, gridDimX, gridDimY, gridDimZ,
                                 blockDimX, blockDimY, blockDimZ,
                                 sharedMemBytes, 0, args.data(), nullptr));
  checkCudaErrors(cuCtxSynchronize());

  for (const auto& param : params) {
    if (param.is_pointer() && !param.is_const()) {
      auto& block = memory::get_block(*reinterpret_cast<uint64_t*>(
          const_cast<char*>(param.data().c_str())));
      auto d_ptr = block.device_ptr(device);
      checkCudaErrors(cuMemcpyDtoH(block.data(), *d_ptr, block.size()));
    }
  }
}

}  // namespace weft::kernel
