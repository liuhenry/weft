project(
  backend
  VERSION 0.1
  LANGUAGES CXX)

# Create backend object library
find_package(Threads REQUIRED)
find_package(Boost REQUIRED)
add_library(backend OBJECT
  device.cc
  kernel.cc
  memory.cc
  server.cc)
target_include_directories(backend PUBLIC
  Boost_INCLUDE_DIRS
  "${CMAKE_SOURCE_DIR}/include"
  "${CMAKE_SOURCE_DIR}/extern/include")
target_link_libraries(backend PUBLIC
  protos)
target_compile_features(backend PUBLIC cxx_std_17)
set_target_properties(backend PROPERTIES CXX_EXTENSIONS OFF)

# Create server executable
add_executable(server main.cc)
target_link_libraries(server PRIVATE
  cuda
  backend)
