#include <iostream>
#include "stress_kernel.cuh"
#include "cupti_profiler.hpp"
#include "cxxopts.hpp" // add to your include path

int main(int argc, char** argv) {
    cxxopts::Options options("gpu_stress_tool", "GPU Stress Testing Utility");
    options.add_options()
        ("t,test", "Test type (memory|compute)", cxxopts::value<std::string>())
        ("i,intensity", "Intensity level", cxxopts::value<int>()->default_value("1"))
        ("h,help", "Print usage");

    auto result = options.parse(argc, argv);
    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    // init_cupti_profiler();
    run_stress_test();
    
    // finalize_cupti_profiler();

    return 0;
}
