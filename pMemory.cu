#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <cuda.h>
#include <cuda_runtime.h>



/*
 * The purpose of this file is to manage the physical memory on the GPU
 */


class pMemory{

    private:
        std::vector<CUmemGenericAllocationHandle> free_handles; // A stack for free handles

    public:

        // Allocates a new xMB frame and pushes it to the stack
        void allocate_push_block(size_t granularity){

                CUmemAllocationProp prop = {};
                prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
                prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                prop.location.id = 0;

                CUmemGenericAllocationHandle handle;
                CUresult  res = cuMemCreate(&handle, granularity, &prop, 0);

                if (res != CUDA_SUCCESS) {
                        const char* errStr;
                        cuGetErrorString(res, &errStr);
                        std::cerr << "CRITICAL CUDA ERROR: cuMemCreate failed with: " << errStr << std::endl;
                        exit(1); // Force the program to stop here so we can see why it failed
                }

                free_handles.push_back(handle);
        }

        // Pushes a handle back onto the free handle stack
        void push_handle(CUmemGenericAllocationHandle handle) {
                free_handles.push_back(handle);
        }
        // Pops a free handle off the stack for use
        CUmemGenericAllocationHandle pop_handle(size_t granularity){
                if (free_handles.empty()){
                        allocate_push_block(granularity);
                }

                CUmemGenericAllocationHandle handle = free_handles.back();
                free_handles.pop_back();
                return handle;
        }

        //TODO: I need something functions that will free the handles at the end (Create a destructor)

};