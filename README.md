# WEFT

## Building

Weft depends on gRPC/Protobuf and builds it from source via `FetchContent` if CMake can't find the config.

The frontend additionally depends on clang (also included via `FetchContent` - LLVM is not installed on the HPC cluster). The LLVM build is configured to only build clang, but this still takes a while, which you can minimize by only calling the Weft build targets:

```
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make server weft
```

## Tests

Tests are taken from the CUDA samples installation. The `CUDA_PATH` in Makefiles may need to be updated for each system. The Weft interposer can then be injected via `LD_PRELOAD`:

```
LD_PRELOAD=/path/to/libweft.so ./runSample
```
