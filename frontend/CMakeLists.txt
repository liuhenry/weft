project(
  frontend
  VERSION 0.1
  LANGUAGES CXX)

find_package(Boost REQUIRED)

# TODO: Allow find/use system libclang
message(STATUS "Using llvm via add_subdirectory (FetchContent).")
include(FetchContent)
FetchContent_Declare(
  llvm
  GIT_REPOSITORY https://github.com/llvm/llvm-project.git
  GIT_TAG        llvmorg-11.0.0
  GIT_PROGRESS   TRUE)
set(FETCHCONTENT_QUIET OFF)
FetchContent_GetProperties(llvm)
if(NOT llvm_POPULATED)
  FetchContent_Populate(llvm)
  set(LLVM_TARGETS_TO_BUILD X86;NVPTX) # no backend so this shouldn't be necessary (but make default target)
  set(LLVM_ENABLE_PIC ON)
  set(LLVM_ENABLE_PROJECTS clang)
  add_subdirectory(${llvm_SOURCE_DIR}/llvm ${llvm_BINARY_DIR})
endif()

add_library(weft SHARED
  client.cc
  libcuhook.cc
  libweft.cc
  nvrtc/kernel_parser.cc)
target_include_directories(weft PUBLIC
  Boost_INCLUDE_DIRS
  "${PROJECT_SOURCE_DIR}"
  "${llvm_SOURCE_DIR}/llvm/include"  # TODO: way to read LLVM_INCLUDE_DIR?
  "${llvm_SOURCE_DIR}/clang/include"
  "${llvm_BINARY_DIR}/include"
  "${llvm_BINARY_DIR}/tools/clang/include")
target_link_libraries(weft PUBLIC
  ${CMAKE_DL_LIBS}
  protos
  libclang
  clangFrontend
  clangTooling)
target_compile_features(weft PUBLIC cxx_std_17)
set_target_properties(weft PROPERTIES CXX_EXTENSIONS OFF)
