#include <iostream>
#include "stress_kernel.cuh"

int main() {
    std::cout << "Launching GPU stress test..." << std::endl;
    run_stress_test();
    return 0;
}
