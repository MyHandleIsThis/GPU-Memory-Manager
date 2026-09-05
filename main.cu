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

    return 0;
}
