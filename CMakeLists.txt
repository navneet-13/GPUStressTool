cmake_minimum_required(VERSION 3.10)
project(GPUStressTool LANGUAGES C CXX CUDA)

cmake_policy(SET CMP0104 NEW)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Set target GPU architectures (adjust to your GPU if needed)
set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)

include_directories(include)
file(GLOB SOURCES src/*.cpp src/*.cu)

add_executable(gpu_stress_tool ${SOURCES})
target_link_libraries(gpu_stress_tool cuda cudart)










