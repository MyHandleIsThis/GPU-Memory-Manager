#include <cuda.h>
#include <cuda_runtime.h>
#include "MemoryManager.cu"

int main() {
    // 1. Initialize the CUDA Driver API (Strictly required first step)
    CUresult resInit = cuInit(0);
    if (resInit != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(resInit, &errStr);
        std::cerr << "cuInit failed: " << errStr << std::endl;
        return -1;
    }

    // 2. Get the Device
    CUdevice device;
    CUresult resDev = cuDeviceGet(&device, 0);
    if (resDev != CUDA_SUCCESS) {
        std::cerr << "Failed to get CUDA device. Is a GPU attached?" << std::endl;
        return -1;
    }

    // 3. Create the Context
    CUcontext context;
    CUresult resCtx = cuCtxCreate(&context, 0, device);
    if (resCtx != CUDA_SUCCESS) {
        std::cerr << "Failed to create CUDA context." << std::endl;
        return -1;
    }

    std::cout << "CUDA Initialized. Instantiating manager..." << std::endl;


    // Instantiate the manager to verify it compiles
    MemoryManager manager;
    std::cout << "Memory Manager compiled and initialized." << std::endl;

    size_t requested_bytes = 1024 * 1024;
    manager.Mem_Alloc(requested_bytes);
    std::cout << "Successfully allocated 1MB." << std::endl;


    // Now it is time to test for logically correctness

    // 1, Testing out read and write
    size_t bytes = sizeof(int);
    CUdeviceptr d_ptr = manager.Mem_Alloc(bytes);
    int host_send = 42;
    int host_recv = 0;

    // Write to GPU and then read it back
    cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(d_ptr), &host_send, bytes);
    cuMemcpyDtoH(&host_recv, reinterpret_cast<CUdeviceptr>(d_ptr), bytes);

    //Verify the correctness
    if (host_recv == host_send) {
        std::cout << "Read/Write Test Passed: Read exactly " << host_recv << "." << std::endl;
    } else {
        std::cerr << "Memory Corruption: Expected " << host_send << " but got " << host_recv << "." << std::endl;
    }

    manager.Mem_Free(d_ptr);

    // 2. Coalescing Test
    size_t block_size = 2 * 1024 * 1024;
    CUdeviceptr ptr1 = manager.Mem_Alloc(block_size);
    CUdeviceptr ptr2 = manager.Mem_Alloc(block_size);
    CUdeviceptr ptr3 = manager.Mem_Alloc(block_size);

    // Freeing all three blocks in a way that allows left + middle + right coalescing
    manager.Mem_Free(ptr1);
    manager.Mem_Free(ptr3);
    manager.Mem_Free(ptr2);

    CUdeviceptr ptr4 = manager.Mem_Alloc(3*block_size);

    if (ptr4 == ptr1) {
        std::cout << "Coalescing Test Passed: Successfully merged and reused the exact same memory." << std::endl;
        manager.Mem_Free(ptr4);
    } else if (ptr4 != 0) {
        std::cerr << "Coalescing Test Failed: Blocks did not merge. Allocator pulled from the giant pool." << std::endl;
        // We got memory, but not from our merged blocks
        manager.Mem_Free(ptr4);
    } else {
        std::cerr << "Allocation entirely failed." << std::endl;
    }

    // 3. Fragmentation Test
    // Free the middle chunk to create a 2MB hole
    size_t chunk_size = 2 * 1024 * 1024; // 2MB
    CUdeviceptr chunkA = manager.Mem_Alloc(chunk_size);
    CUdeviceptr chunkB = manager.Mem_Alloc(chunk_size);
    CUdeviceptr chunkC = manager.Mem_Alloc(chunk_size);
    manager.Mem_Free(chunkB);

    // Request a 1MB block (assuming granularity is <= 1MB)
    // It should fit perfectly inside the old chunkB address space.
    CUdeviceptr split_chunk = manager.Mem_Alloc(1024 * 1024);

    if (split_chunk == chunkB) {
        std::cout << "Fragmentation Test Passed: Allocator reused and split the hole." << std::endl;
    } else {
        std::cerr << "Fragmentation Test Failed: Allocator ignored the hole." << std::endl;
    }

    manager.Mem_Free(chunkA);
    manager.Mem_Free(chunkC);
    manager.Mem_Free(split_chunk); // Note: We only free the 1MB we took!


    return 0;
}
