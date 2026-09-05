#include <iostream>
#include <string>
#include <map>
#include <cuda.h>
#include <cuda_runtime.h>
#include <optional>

/*
 * The purpose of this file is to manage the virtual memory of the memory manager
 */
class vMemory {

    private:
        CUdeviceptr base_address;// Base address of the VM TODO: I probably need a getter / setter method for this
        size_t max_size;

        std::multimap<size_t, CUdeviceptr> free_pages_by_size; // All pages that are not allocated
        std::map<CUdeviceptr, size_t> free_pages_by_address;
        std::map<CUdeviceptr, size_t> active_pages; // Pages that are allocated


        // Return the address of the new split block
        CUdeviceptr split_page(CUdeviceptr address, size_t size){
                return address + size;
        }

        void coalesce_pages(CUdeviceptr address){

                auto middle_page = free_pages_by_address.find(address);

                if (middle_page == free_pages_by_address.end()) {
                        return;
                }


                auto left_page = middle_page;
                auto right_page = std::next(middle_page);

                // Checking to see if a left page and right page exists
                bool has_left = middle_page != free_pages_by_address.begin();
                if (has_left){
                        --left_page;
                }

                bool has_right = right_page != free_pages_by_address.end();

                // Checking continuity between the pages
                bool left_contiguous = has_left && left_page->first + left_page->second == address;
                bool right_contiguous = has_right && address + middle_page->second == right_page->first;

                if (left_contiguous && right_contiguous){ // left + middle + right merging

                        CUdeviceptr left_address = left_page->first;
                        CUdeviceptr right_address = right_page->first;
                        size_t left_size = free_pages_by_address[left_address];
                        size_t right_size = free_pages_by_address[right_address];
                        size_t middle_size = free_pages_by_address[address];

                        free_pages_by_address.erase(left_address);
                        free_pages_by_address.erase(right_address);
                        free_pages_by_address.erase(address);

                        auto range_left = free_pages_by_size.equal_range(left_size);
                        for (auto it = range_left.first; it != range_left.second; it++){
                        if (it->second == left_address) {
                                free_pages_by_size.erase(it);
                                break;
                                }
                        }

                        auto range_right = free_pages_by_size.equal_range(right_size);
                        for (auto it = range_right.first; it != range_right.second; it++){
                        if (it->second == right_address) {
                                free_pages_by_size.erase(it);
                                break;
                                }
                        }

                        auto range_middle = free_pages_by_size.equal_range(middle_size);
                        for (auto it = range_middle.first; it != range_middle.second; it++){
                        if (it->second == address) {
                                free_pages_by_size.erase(it);
                                break;
                                }
                        }

                        free_pages_by_address[left_address] = left_size + middle_size + right_size;
                        free_pages_by_size.insert({left_size + middle_size + right_size, left_address});


                } else if (left_contiguous){ // left + middle merging
                        CUdeviceptr left_address = left_page->first;
                        size_t left_size = free_pages_by_address[left_address];
                        size_t middle_size = free_pages_by_address[address];

                        free_pages_by_address.erase(left_address);
                        free_pages_by_address.erase(address);

                        auto range_left = free_pages_by_size.equal_range(left_size);
                        for (auto it = range_left.first; it != range_left.second; it++){
                                if (it->second == left_address) {
                                        free_pages_by_size.erase(it);
                                        break;
                                }
                        }

                        auto range_middle = free_pages_by_size.equal_range(middle_size);
                        for (auto it = range_middle.first; it != range_middle.second; it++){
                                if (it->second == address) {
                                        free_pages_by_size.erase(it);
                                        break;
                                }
                        }

                        free_pages_by_address[left_address] = left_size + middle_size;
                        free_pages_by_size.insert({left_size + middle_size, left_address});



                } else if (right_contiguous){ // middle + right merging
                        CUdeviceptr right_address = right_page->first;
                        size_t right_size = free_pages_by_address[right_address];
                        size_t middle_size = free_pages_by_address[address];

                        free_pages_by_address.erase(right_address);
                        free_pages_by_address.erase(address);

                        auto range_right = free_pages_by_size.equal_range(right_size);
                        for (auto it = range_right.first; it != range_right.second; it++){
                                if (it->second == right_address) {
                                        free_pages_by_size.erase(it);
                                        break;
                                }
                        }

                        auto range_middle = free_pages_by_size.equal_range(middle_size);
                        for (auto it = range_middle.first; it != range_middle.second; it++){
                                if (it->second == address) {
                                        free_pages_by_size.erase(it);
                                        break;
                                }
                        }

                        free_pages_by_address[address] = middle_size + right_size;
                        free_pages_by_size.insert({middle_size + right_size, address});




                } else { // No merging
                        return;
                }



        }


    public:

        vMemory(CUdeviceptr address, size_t size){
                base_address = address;
                max_size = size;
                free_pages_by_size.insert({size, address});
                free_pages_by_address[address] = size;
        }




    // Find the smallest free page that is greater than or equal to requested size
        // We will assume that requested_size is aligned to the granularity
        std::optional<CUdeviceptr> reserve_region(size_t requested_size){

                if (requested_size > max_size){
                        return std::nullopt;
                }

                auto pair = free_pages_by_size.lower_bound(requested_size);

                if (pair == free_pages_by_size.end()) { // No valid page was found
                        return std::nullopt;
                }

                size_t size_of_page = pair->first;
                CUdeviceptr virtual_address = pair->second;

                // Removing the page from the free_pages
                free_pages_by_size.erase(pair);
                free_pages_by_address.erase(virtual_address);

                // Determining the block to give the user
                if (size_of_page == requested_size){
                        // No need to split since we found a page of the exact size we need
                        active_pages[virtual_address] = requested_size;
                        return virtual_address;

                }  else {
                        // Split the page
                        CUdeviceptr new_va = split_page(virtual_address, requested_size); // Returns address of the new split block
                        active_pages[virtual_address] = requested_size;
                        free_pages_by_size.insert({size_of_page - requested_size,  new_va});
                        free_pages_by_address[new_va] = size_of_page - requested_size ;
                        return virtual_address;
                }
        }

        // Moves an address from active_pages back to free_pages, and merges adjacent pages
        size_t free_region(size_t address){
                size_t size = active_pages[address];
                free_pages_by_address[address] = size;
                free_pages_by_size.insert({size, address});
                active_pages.erase(address);
                coalesce_pages(address);
                return size;
        }
};



