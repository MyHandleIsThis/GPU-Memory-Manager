#include "MemoryManager.cu"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <iostream>


static MemoryManager* manager = nullptr;

extern "C" {
void* my_malloc(ssize_t size, int device, cudaStream_t stream) {
    // PyTorch passes signed sizes; cast to size_t for your manager
    if (!manager){
        CUresult res = cuInit(0);
        cudaSetDevice(device);
        manager = new MemoryManager();
    }

    CUdeviceptr ptr = manager->Mem_Alloc(static_cast<size_t>(size));
    return reinterpret_cast<void*>(ptr);
}

void my_free(void* ptr, ssize_t size, int device, cudaStream_t stream) {

    if (manager && ptr != nullptr) {
        cudaDeviceSynchronize();
        manager->Mem_Free(reinterpret_cast<CUdeviceptr>(ptr));
    }
}
}
