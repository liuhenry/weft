#include "kernel.h"

#include <cuda.h>

#include <boost/range/adaptor/indexed.hpp>
#include <functional>
#include <random>
#include <unordered_map>

#include "CUDA_samples/helper_cuda_drvapi.h"
#include "memory.h"
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

void Function::execute(Device& device, const ExecutionArgs execution) const {
  checkCudaErrors(cuCtxSetCurrent(device));

  device.stream_pool.consume_one([&](const CUstream& stream) -> void {
    std::clog << "> LaunchKernel: " << this->handle() << " (" << this->name()
              << ") on Device: " << device << ", Stream: " << stream << "\n"
              << "\t Grid — X: " << execution.gridDimX
              << ", Y: " << execution.gridDimY << ", Z: " << execution.gridDimZ
              << "\n"
              << "\t Block — X: " << execution.blockDimX
              << ", Y: " << execution.blockDimY
              << ", Z: " << execution.blockDimZ << "\n"
              << "\t Shared Memory: " << execution.sharedMemBytes << "\n";

    CUmodule module;
    checkCudaErrors(
        cuModuleLoadDataEx(&module, this->module().ptx().c_str(), 0, 0, 0));
    CUfunction kernel_addr;
    checkCudaErrors(
        cuModuleGetFunction(&kernel_addr, module, this->name().c_str()));

    std::vector<void*> args;
    for (const auto& param : execution.args | boost::adaptors::indexed(0)) {
      void* ptr;
      if (param.value().is_pointer()) {
        auto& block = memory::get_block(*reinterpret_cast<uint64_t*>(
            const_cast<char*>(param.value().data().c_str())));
        auto d_ptr = block.device_ptr(device);  // Also cuMalloc's if we haven't

        // NOTE: we'll also copy the const pointed data
        // This allows us to perform the later consistency check on
        // read-back if (param.value().is_const()) {
        checkCudaErrors(
            cuMemcpyHtoDAsync(*d_ptr, block.data(), block.size(), stream));
        // }
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

    // Add _weft_blockOffset (assume last)
    args.emplace_back(const_cast<int*>(&execution.blockOffset));

    checkCudaErrors(cuLaunchKernel(
        kernel_addr, execution.gridDimX, execution.gridDimY, execution.gridDimZ,
        execution.blockDimX, execution.blockDimY, execution.blockDimZ,
        execution.sharedMemBytes, stream, args.data(), nullptr));

    for (const auto& param : execution.args) {
      if (param.is_pointer() && !param.is_const()) {
        memory::get_block(*reinterpret_cast<uint64_t*>(
                              const_cast<char*>(param.data().c_str())))
            .write_back(device, stream);
      }
    }
  });
}

}  // namespace weft::kernel
