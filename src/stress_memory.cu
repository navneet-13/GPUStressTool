#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <nvml.h>
#include <vector>
#include <chrono>

__global__ void mem_copy_kernel(float* dst, const float* src, size_t N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] = src[i];
}



void print_memory_utilization() {
    nvmlInit();
    nvmlDevice_t dev;
    nvmlDeviceGetHandleByIndex(0, &dev);
    nvmlUtilization_t utilization;
    nvmlDeviceGetUtilizationRates(dev, &utilization);
    std::cout << "[NVML] GPU Util: " << utilization.gpu << "%, Mem Util: " << utilization.memory << "%\n";
    nvmlShutdown();
}

void stress_h2d_d2h_bandwidth(size_t N, int repetitions) {
    float *h_data, *d_data;
    cudaMallocHost(&h_data, N * sizeof(float));  // pinned host memory
    cudaMalloc(&d_data, N * sizeof(float));

    for (int i = 0; i < N; ++i) h_data[i] = 1.0f;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < repetitions; ++i) {
        cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    double total_bytes = 2.0 * repetitions * N * sizeof(float); // H2D + D2H
    double time_sec = std::chrono::duration<double>(end - start).count();
    double bandwidth_GBps = total_bytes / (time_sec * 1e9);

    std::cout << "[H2D+D2H] " << bandwidth_GBps << " GB/s over " << repetitions << " repetitions\n";

    cudaFreeHost(h_data);
    cudaFree(d_data);
}

void stress_d2d_bandwidth(size_t N, int repetitions) {
    float *src, *dst;
    cudaMalloc(&src, N * sizeof(float));
    cudaMalloc(&dst, N * sizeof(float));
    cudaMemset(src, 1, N * sizeof(float));

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < repetitions; ++i) {
        cudaMemcpy(dst, src, N * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    double total_bytes = repetitions * N * sizeof(float);
    double time_sec = std::chrono::duration<double>(end - start).count();
    double bandwidth_GBps = total_bytes / (time_sec * 1e9);

    std::cout << "[D2D] " << bandwidth_GBps << " GB/s over " << repetitions << " repetitions\n";

    cudaFree(src);
    cudaFree(dst);
}


void stress_async_streamed_h2d_d2h(size_t N, int reps, int num_streams) {
    std::vector<cudaStream_t> streams(num_streams);
    float *h_data[num_streams], *d_data[num_streams];

    for (int s = 0; s < num_streams; ++s) {
        cudaStreamCreate(&streams[s]);
        cudaMallocHost(&h_data[s], N * sizeof(float));
        cudaMalloc(&d_data[s], N * sizeof(float));
        std::fill(h_data[s], h_data[s] + N, 1.0f);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < reps; ++i) {
        for (int s = 0; s < num_streams; ++s) {
            cudaMemcpyAsync(d_data[s], h_data[s], N * sizeof(float), cudaMemcpyHostToDevice, streams[s]);
            cudaMemcpyAsync(h_data[s], d_data[s], N * sizeof(float), cudaMemcpyDeviceToHost, streams[s]);
        }
    }

    for (int s = 0; s < num_streams; ++s) {
        cudaStreamSynchronize(streams[s]);
    }

    auto end = std::chrono::high_resolution_clock::now();

    double total_bytes = 2.0 * reps * num_streams * N * sizeof(float);
    double time_sec = std::chrono::duration<double>(end - start).count();
    double bandwidth = total_bytes / (time_sec * 1e9);

    std::cout << "[Async H2D+D2H] " << bandwidth << " GB/s with " << num_streams << " streams\n";

    for (int s = 0; s < num_streams; ++s) {
        cudaStreamDestroy(streams[s]);
        cudaFree(d_data[s]);
        cudaFreeHost(h_data[s]);
    }
}


void stress_nvlink_peer_copy(size_t N, int reps) {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount < 2) {
        std::cerr << "[NVLink] Skipping peer test — only one GPU found.\n";
        return;
    }

    cudaSetDevice(0);
    float* d0; cudaMalloc(&d0, N * sizeof(float));
    cudaSetDevice(1);
    float* d1; cudaMalloc(&d1, N * sizeof(float));

    cudaDeviceEnablePeerAccess(0, 0);
    cudaDeviceEnablePeerAccess(1, 0);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < reps; ++i) {
        cudaMemcpyPeer(d0, 0, d1, 1, N * sizeof(float));
        cudaMemcpyPeer(d1, 1, d0, 0, N * sizeof(float));
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    double total_bytes = 2.0 * reps * N * sizeof(float);
    double time_sec = std::chrono::duration<double>(end - start).count();
    double bandwidth = total_bytes / (time_sec * 1e9);

    std::cout << "[NVLink P2P] " << bandwidth << " GB/s\n";

    cudaFree(d0);
    cudaFree(d1);
}

void run_stress_test() {
    size_t N = 1 << 24;
    float *d_src, *d_dst;
    cudaMalloc(&d_src, N * sizeof(float));
    cudaMalloc(&d_dst, N * sizeof(float));

    auto start = std::chrono::high_resolution_clock::now();
    mem_copy_kernel<<<(N + 255) / 256, 256>>>(d_dst, d_src, N);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    std::ofstream log("logs/stress_log.csv", std::ios::app);
    std::cout << "memory," << N << "," << std::chrono::duration<double, std::milli>(end - start).count() << std::endl;
    // fflush(stdout);
    log.close();
    

    std::cout << "Memory stress test completed." << std::endl;
    cudaFree(d_src);
    cudaFree(d_dst);

    N = 1 << 26; // 64MB
    const int reps = 100;

    std::cout << "--- Running Bandwidth Stress Tests ---\n";
    stress_h2d_d2h_bandwidth(N, reps);
    std::cout << "--- Running D2D Bandwidth Stress Test ---\n";
    stress_d2d_bandwidth(N, reps);
    std::cout << "--- Running Async Streamed H2D+D2H Stress Test ---\n";
    stress_async_streamed_h2d_d2h(N, reps, 40);
    std::cout << "--- Running NVLink Peer Copy Stress Test ---\n";
    stress_nvlink_peer_copy(N, 100*reps);

    print_memory_utilization();
}
