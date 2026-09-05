#include <iostream>
#include <string>
#include <map>
#include <vector>
#include "vMemory.cu"
#include "pMemory.cu"
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>



/*
 *
 */

class MemoryManager{

    private:

        std::map<CUdeviceptr, std::vector<CUmemGenericAllocationHandle>> allocated_handles; // A stack for allocated_handles;

        pMemory pManager;
        vMemory vManager = vMemory(0,0);
        CUdeviceptr base_address;
        size_t granularity;

        CUmemAccessDesc accessDesc{};
        CUmemAllocationProp prop = {};

public:

        MemoryManager(){

            accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            accessDesc.location.id = 0;
            accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = 0;

            cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
            cuMemAddressReserve(&base_address, granularity * 2000, 0, 0, 0);

            // Now initalize the virtual function with granularity and base address
            vManager = vMemory(base_address, granularity * 2000);
        }

        // Allocates requested_size bytes and returns a pointer to the allocated memory
        void Mem_Alloc(size_t requested_size){
            // Round up requested_size to a multiple of the granularity and reserve the VM for it
            size_t size = ((requested_size + granularity - 1) / granularity) * granularity;
            auto base_check = vManager.reserve_region(size);

            if (!base_check){
                return;
            }

            // Mapping the VM to PM and setting permissions
            CUdeviceptr base = base_check.value();
            for (size_t i = 0; i < size / granularity; i++){
                CUdeviceptr address = *base + i * granularity;
                CUmemGenericAllocationHandle handle = pManager.pop_handle(granularity);
                cuMemMap(address, granularity, 0, handle, 0 );
                cuMemSetAccess(address, granularity, &accessDesc, 1);
                allocated_handles[*base].push_back(handle);

            }
        }

        // Frees the memory space pointed to by ptr
        void Mem_Free(CUdeviceptr ptr){
            auto it = allocated_handles.find(ptr);
            if (it == allocated_handles.end()) {
                // Invalid allocation.
                return;
            }

            size_t size = vManager.free_region(ptr);
            std::vector<CUmemGenericAllocationHandle> handles = it->second;
            CUresult res = cuMemUnmap(ptr, size);

            for (const auto& handle: handles){
                pManager.push_handle(handle);
            }

            allocated_handles.erase(it);
        }



};
