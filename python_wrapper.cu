#include "MemoryManager.cu"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <iostream>


static MemoryManager manager = MemoryManager();

extern "C" {
    void* my_malloc(ssize_t size, int device, cudaStream_t stream) {
        // PyTorch passes signed sizes; cast to size_t for your manager
        std::cout << "Allocating Memory!" << std::endl;
        CUdeviceptr ptr = manager.Mem_Alloc(static_cast<size_t>(size));
        return reinterpret_cast<void*>(ptr);
    }

    void my_free(void* ptr, ssize_t size, int device, cudaStream_t stream) {
        std::cout << "Freeing Memory" << std::endl;

        if (ptr != nullptr) {
            manager.Mem_Free(reinterpret_cast<CUdeviceptr>(ptr));
        }
    }
}
