#define _GNU_SOURCE
#include <cuda.h>
#include <dlfcn.h>
#include <grpcpp/channel.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
#include <nvrtc.h>
#include <stdio.h>
#include <unistd.h>

#include <iomanip>
#include <iostream>
#include <string_view>

#include "client.h"
#include "libcuhook.h"
#include "nvrtc/kernel_parser.h"

// Helper function to run initialization steps
#define ASSERT_COND(x, msg)                                                    \
  do {                                                                         \
    if (!(x)) {                                                                \
      fprintf(stderr, "Error: Condition (%s) failed at %s:%d\n", #x, __FILE__, \
              __LINE__);                                                       \
      fprintf(stderr, "cuHook load failed (%s)\n", msg);                       \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static weft::CudaDriverClient client;
static weft::nvrtc::KernelMetadata metadata;
static bool weftInitialized = false;

CUresult MemAlloc_intercept(CUdeviceptr *dptr, size_t bytesize) {
  std::clog << "* " << std::setw(6) << getpid()
            << " >> Received cuMemAlloc! Loc: " << dptr << " for " << bytesize
            << "\n";
  *dptr = client.MemAlloc(bytesize);
  std::clog << "* " << std::setw(6) << getpid() << " >> Handle: " << *dptr
            << "\n";
  return CUDA_SUCCESS;
}

CUresult MemFree_intercept(CUdeviceptr dptr) {
  std::clog << "* " << std::setw(6) << getpid()
            << " >> Received cuMemFree! Handle: " << dptr << "\n";
  client.MemFree(dptr);
  return CUDA_SUCCESS;
}

CUresult MemcpyHtoD_intercept(CUdeviceptr dstDevice, const void *srcHost,
                              size_t ByteCount) {
  std::clog << "* " << std::setw(6) << getpid()
            << " >> Received cuMemcpyHtoD! Handle: " << dstDevice
            << " from: " << srcHost << " for " << ByteCount << "\n";
  std::string_view srcHostView(static_cast<const char *>(srcHost), ByteCount);
  client.MemcpyHtoD(dstDevice, srcHostView);
  return CUDA_SUCCESS;
}

CUresult MemcpyDtoH_intercept(void *dstHost, CUdeviceptr srcDevice,
                              size_t ByteCount) {
  std::clog << "* " << std::setw(6) << getpid()
            << " >> Received cuMemcpyDtoH! Dest: " << dstHost
            << " from: " << srcDevice << " for " << ByteCount << "\n";
  // FIXME: Can we do a zero-copy with return semantics?
  client.MemcpyDtoH(dstHost, srcDevice, ByteCount);
  return CUDA_SUCCESS;
}

CUresult ModuleGetFunction_intercept(CUfunction *hfunc, CUmodule hmod,
                                     const char *name) {
  auto m_handle = reinterpret_cast<uint64_t>(hmod);
  std::clog << "* " << std::setw(6) << getpid()
            << " >> Received cuModuleGetFunction!\n\tModule: " << m_handle
            << ", name: " << name << "\n";

  auto f_handle = client.ModuleGetFunction(m_handle, name, *metadata.at(name));

  // FIXME: this cast is terrible...
  *hfunc = reinterpret_cast<CUfunction>(f_handle);
  metadata.emplace(f_handle, metadata.at(name));

  std::clog << "\tAssociated with " << f_handle << "\n";
  return CUDA_SUCCESS;
}

CUresult ModuleLoadDataEx_intercept(CUmodule *module, const void *image,
                                    uint32_t numOptions, CUjit_option *options,
                                    void *optionValues[]) {
  std::clog << "* " << std::setw(6) << getpid()
            << " >> Received cuModuleLoadDataEx!\n";
  // TODO: Support non-string image (file/executable resource)
  // TODO: Support options

  // FIXME: this cast is also terrible...
  *module = reinterpret_cast<CUmodule>(
      client.ModuleLoadData(reinterpret_cast<const char *>(image)));
  return CUDA_SUCCESS;
}

CUresult LaunchKernel_intercept(CUfunction f, uint32_t gridDimX,
                                uint32_t gridDimY, uint32_t gridDimZ,
                                uint32_t blockDimX, uint32_t blockDimY,
                                uint32_t blockDimZ, uint32_t sharedMemBytes,
                                CUstream hStream, void *kernelParams[],
                                void *extra[]) {
  std::clog << "* " << std::setw(6) << getpid()
            << " >> Received cuLaunchKernel!"
            << "\n"
            << "\tGrid — X: " << gridDimX << ", Y: " << gridDimY
            << ", Z: " << gridDimZ << "\n"
            << "\tBlock — X: " << blockDimX << ", Y: " << blockDimY
            << ", Z: " << blockDimZ << "\n"
            << "\tShared Memory: " << sharedMemBytes << ", Stream: " << hStream
            << "\n";

  if (extra) {
    // TODO: hStream also unsupported
    std::cerr << "Error: passing extra is unsupported!\n";
  }

  auto handle = reinterpret_cast<uint64_t>(f);
  client.LaunchKernel(handle, gridDimX, gridDimY, gridDimZ, blockDimX,
                      blockDimY, blockDimZ, sharedMemBytes,
                      reinterpret_cast<uint64_t>(hStream), *metadata.at(handle),
                      kernelParams);
  return CUDA_SUCCESS;
}

void weft_init() {
  // Load the cudaHookRegisterCallback symbol using the default library search
  // order. If we found the symbol, then the hooking library has been loaded
  auto cuHook = reinterpret_cast<fnCuHookRegisterCallback>(
      dlsym(RTLD_DEFAULT, "cuHookRegisterCallback"));
  ASSERT_COND(cuHook, dlerror());
  if (cuHook) {
    std::clog << "* " << std::setw(6) << getpid()
              << " >> CUDA WEFT Frontend loaded.\n";

    client = weft::CudaDriverClient{grpc::CreateChannel(
        "localhost:50051", grpc::InsecureChannelCredentials())};

    // CUDA Runtime symbols cannot be hooked but the underlying driver ones
    // _can_. Example:
    // - cudaFree() will trigger cuMemFree
    // - cudaDeviceReset() will trigger a context change and you would need to
    // intercept cuCtxGetCurrent/cuCtxSetCurrent
    cuHook(CU_HOOK_MEM_ALLOC, INTERCEPT_HOOK,
           reinterpret_cast<void *>(MemAlloc_intercept));
    cuHook(CU_HOOK_MEM_FREE, INTERCEPT_HOOK,
           reinterpret_cast<void *>(MemFree_intercept));
    cuHook(CU_HOOK_MEMCPY_H_TO_D, INTERCEPT_HOOK,
           reinterpret_cast<void *>(MemcpyHtoD_intercept));
    cuHook(CU_HOOK_MEMCPY_D_TO_H, INTERCEPT_HOOK,
           reinterpret_cast<void *>(MemcpyDtoH_intercept));
    cuHook(CU_HOOK_MODULE_GET_FUNCTION, INTERCEPT_HOOK,
           reinterpret_cast<void *>(ModuleGetFunction_intercept));
    cuHook(CU_HOOK_MODULE_LOAD_DATA_EX, INTERCEPT_HOOK,
           reinterpret_cast<void *>(ModuleLoadDataEx_intercept));
    cuHook(CU_HOOK_LAUNCH_KERNEL, INTERCEPT_HOOK,
           reinterpret_cast<void *>(LaunchKernel_intercept));
    weftInitialized = true;
  }
}

// Direct LD_PRELOAD overrides

CUresult cuInit(uint32_t Flags) {
  if (!weftInitialized) weft_init();
  // FIXME: should we use real_dlsym from libcuhook instead?
  return reinterpret_cast<CUresult (*)(uint32_t)>(dlsym(RTLD_NEXT, "cuInit"))(
      Flags);
}

nvrtcResult nvrtcCreateProgram(nvrtcProgram *prog, const char *src,
                               const char *name, int numHeaders,
                               const char *const *headers,
                               const char *const *includeNames) {
  std::clog << "nvrtcCreateProgram:\n";
  metadata.parse_cu(src);

  typedef nvrtcResult (*fnNvrtcCreateProgram)(
      nvrtcProgram *, const char *, const char *, int, const char *const *,
      const char *const *);
  return reinterpret_cast<fnNvrtcCreateProgram>(
      dlsym(RTLD_NEXT, "nvrtcCreateProgram"))(prog, src, name, numHeaders,
                                              headers, includeNames);
}
