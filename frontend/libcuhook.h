#ifndef _CUHOOK_H_
#define _CUHOOK_H_

#include <cuda.h>

typedef enum CuHookTypesEnum {
  PRE_CALL_HOOK,
  INTERCEPT_HOOK,
  POST_CALL_HOOK,
  CU_HOOK_TYPES,
} CuHookTypes;

typedef enum CuHookSymbolsEnum {
  CU_HOOK_MEM_ALLOC,
  CU_HOOK_MEM_FREE,
  CU_HOOK_MEMCPY_H_TO_D,
  CU_HOOK_MEMCPY_D_TO_H,
  CU_HOOK_CTX_GET_CURRENT,
  CU_HOOK_CTX_SET_CURRENT,
  CU_HOOK_CTX_DESTROY,
  CU_HOOK_MODULE_GET_FUNCTION,
  CU_HOOK_MODULE_LOAD_DATA_EX,
  CU_HOOK_LAUNCH_KERNEL,
  CU_HOOK_SYMBOLS,
} CuHookSymbols;

// One and only function to call to register a callback
// You need to dlsym this symbol in your application and call it to register
// callbacks
typedef void (*fnCuHookRegisterCallback)(CuHookSymbols symbol, CuHookTypes type,
                                         void *callback);
typedef void *(*fnCuHookRealFunc)(CuHookSymbols symbol);
extern "C" {
void cuHookRegisterCallback(CuHookSymbols symbol, CuHookTypes type,
                            void *callback);
void *cuHookRealFunc(CuHookSymbols symbol);
}

// In case you want to intercept, the callbacks need the same type/parameters as
// the real functions
typedef CUresult CUDAAPI (*fnMemAlloc)(CUdeviceptr *dptr, size_t bytesize);
typedef CUresult CUDAAPI (*fnMemFree)(CUdeviceptr dptr);
typedef CUresult CUDAAPI (*fnMemcpyHtoD)(CUdeviceptr dstDevice,
                                         const void *srcHost, size_t ByteCount);
typedef CUresult CUDAAPI (*fnMemcpyDtoH)(void *dstHost, CUdeviceptr srcDevice,
                                         size_t ByteCount);

typedef CUresult CUDAAPI (*fnCtxGetCurrent)(CUcontext *pctx);
typedef CUresult CUDAAPI (*fnCtxSetCurrent)(CUcontext ctx);
typedef CUresult CUDAAPI (*fnCtxDestroy)(CUcontext ctx);

typedef CUresult CUDAAPI (*fnModuleLoadDataEx)(CUmodule *module,
                                               const void *image,
                                               uint32_t numOptions,
                                               CUjit_option *options,
                                               void *optionValues[]);
typedef CUresult CUDAAPI (*fnLaunchKernel)(
    CUfunction f, uint32_t gridDimX, uint32_t gridDimY, uint32_t gridDimZ,
    uint32_t blockDimX, uint32_t blockDimY, uint32_t blockDimZ,
    uint32_t sharedMemBytes, CUstream hStream, void *kernelParams[],
    void *extra[]);

#endif /* _CUHOOK_H_ */
