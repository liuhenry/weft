cmake_minimum_required(VERSION 3.15)

project(
    Weft
    LANGUAGES CUDA CXX)

add_subdirectory(protos)
add_subdirectory(backend)
add_subdirectory(frontend)
