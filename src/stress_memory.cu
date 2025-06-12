#include <cuda_runtime.h>
#include <iostream>

__global__ void mem_copy_kernel(float* dst, const float* src, size_t N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] = src[i];
}

void run_stress_test() {
    const size_t N = 1 << 24;
    float *d_src, *d_dst;
    cudaMalloc(&d_src, N * sizeof(float));
    cudaMalloc(&d_dst, N * sizeof(float));

    mem_copy_kernel<<<(N + 255) / 256, 256>>>(d_dst, d_src, N);
    cudaDeviceSynchronize();

    std::cout << "Memory stress test completed." << std::endl;
    cudaFree(d_src);
    cudaFree(d_dst);
}
