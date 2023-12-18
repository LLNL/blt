#include <mpi.h>
#include <iostream>
#include <ostream>
#include <string>

int main() {
    int rank = 0;
    std::cout << "Hello from host ";
    std::cout << rank;
    std::cout << std::endl;
}
