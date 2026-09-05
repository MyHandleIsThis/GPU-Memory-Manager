#include <cuda.h>
#include "MemoryManager.cu"

int main() {
    // Instantiate the manager to verify it compiles
    MemoryManager manager;
    std::cout << "Memory Manager compiled and initialized." << std::endl;

    size_t requested_bytes = 1024 * 1024;
    manager.Mem_Alloc(requested_bytes);
    std::cout << "Successfully allocated 1MB." << std::endl;

    return 0;
}
