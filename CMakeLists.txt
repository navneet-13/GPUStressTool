cmake_minimum_required(VERSION 3.10)
project(GPUStressTool LANGUAGES C CXX CUDA)

cmake_policy(SET CMP0104 NEW)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Target architectures for H100, A100, H200
set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)

# Include headers
include_directories(
    include
    $ENV{CUDA_HOME}/targets/x86_64-linux/include
)

# Source files
file(GLOB SOURCES src/*.cpp src/*.cu)

# Add executable target
add_executable(gpu_stress_tool ${SOURCES})

# Locate CUPTI and NVML libraries
find_library(CUPTI_LIBRARY cupti HINTS $ENV{CUDA_HOME}/lib64)
find_library(NVML_LIBRARY nvidia-ml HINTS /usr/lib64 /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64)

# Link all required libraries in one call
target_link_libraries(gpu_stress_tool
    cuda
    cudart
    ${CUPTI_LIBRARY}
    ${NVML_LIBRARY}
)
