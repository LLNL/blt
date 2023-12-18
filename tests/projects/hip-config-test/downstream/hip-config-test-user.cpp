#include <iostream>

#include "hip-library.hpp"

int main() {
    std::cout << "Hello from downstream project.  Running hip hello world: " << std::endl;
    run_kernel();

    return 0;
}
