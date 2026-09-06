# cuMemManager: Unified CUDA Virtual Memory Allocator for PyTorch
A custom GPU memory allocator built in C++ using the low-level CUDA Driver API. This project implements a unified virtual memory manager designed to be dynamically injected into machine learning frameworks (like PyTorch and CuPy) via the Pluggable Allocator API.

By taking direct control of virtual address reservations and physical page mapping, this allocator breaks down the siloed memory pools of independent libraries, allowing physical GPU memory to be safely and seamlessly recycled across framework boundaries to prevent out-of-memory (OOM) faults.

## 🚀 Key Features
Low-Level Driver API Integration: Bypasses standard CUDA Runtime constraints using cuMemCreate, cuMemAddressReserve, and cuMemMap for precise physical page mapping.

O(logN) Free-Page Coalescing: Implements dynamic block splitting and adjacent free-page merging using std::map, strictly controlling virtual address space fragmentation during volatile tensor lifecycles.

Cross-Framework Unification: Exposes the C++ manager through a C-compatible shared library (.so), allowing injection into PyTorch via CUDAPluggableAllocator and CuPy via custom allocation routing.

OOM Prevention: Dynamically reclaims and remaps physical pages from discarded arrays directly into autograd computational graphs, drastically reducing peak VRAM requirements.

## 📊 The Problem vs. The Solution
The Problem (Siloed Memory Hoarding):
In standard pipelines, libraries like CuPy and PyTorch maintain independent caching allocators. When CuPy finishes preprocessing data, it marks the memory as free internally but does not return the physical pages to the GPU. When PyTorch begins model training, it is locked out of those idle pages and must request brand new physical VRAM. This stacking effect artificially inflates peak VRAM usage and triggers OOM crashes on large datasets.

## The Solution:
cuMemManager acts as a single, unified memory pool. When CuPy deletes a tensor, the C++ manager physically unmaps the pages using cuMemUnmap. When PyTorch subsequently requests memory, those exact physical pages are immediately remapped into PyTorch's virtual address space.

Benchmark Results: End-to-End Classification Pipeline
Test conditions: 20-million-row synthetic tabular dataset, preprocessing in CuPy -> MLP training in PyTorch.

Metric	Default Allocators (Siloed)	cuMemManager (Unified)	Difference
Peak VRAM Usage	9.63 GB	4.91 GB	- 49%
Execution Time	19.33 s	39.46 s	+ 104%
Pipeline Stability	Frequent OOMs	Stable	-
Engineering Trade-offs: The 49% reduction in physical VRAM comes at the cost of increased execution time. To safely unmap physical pages during the framework handoff without triggering CUDA_ERROR_ILLEGAL_ADDRESS faults from asynchronous queues, the custom allocator enforces strict stream synchronization (cudaDeviceSynchronize()). This project deliberately trades execution speed for massive gains in effective memory capacity.

## 🛠️ Architecture
Virtual Address Pool: Upon initialization, the allocator reserves a massive contiguous block of virtual address space (e.g., 100 GB) via cuMemAddressReserve. This costs zero physical RAM.

Physical Allocation: When memory is requested, physical pages are instantiated via cuMemCreate at a defined granularity (e.g., 2MB) and mapped to the virtual addresses via cuMemMap.

Defragmentation: A C++ std::map tracks available address ranges. Freed blocks are checked against neighboring virtual addresses and coalesced in O(logN) time to prevent fragmentation.

Python Wrapper: The logic is exposed via my_malloc and my_free functions enclosed in an extern "C" block, compiled as a shared object, and loaded by PyTorch's ctypes backend.

💻 Getting Started
1. Compilation
Ensure you have the CUDA Toolkit installed. Compile the C++ source into a shared library:

Bash
nvcc -std=c++17 -Xcompiler -fPIC -shared allocator_wrapper.cu -o custom_allocator.so -lcuda
2. Usage in PyTorch
You can inject the allocator programmatically in your Python scripts:

Python
import torch
import os

### Resolve path to the compiled shared object
so_path = os.path.abspath('custom_allocator.so')

### Register and activate the pluggable allocator
new_alloc = torch.cuda.memory.CUDAPluggableAllocator(so_path, 'my_malloc', 'my_free')
torch.cuda.memory.change_current_allocator(new_alloc)

### All subsequent PyTorch allocations will now route to cuMemManager
tensor = torch.ones((1024, 1024), device='cuda')
Alternatively, you can set it globally via environment variables before running your script:

### Run baseline (Siloed memory pools)
python tests/benchmark.py
### Run unified allocator
python tests/benchmark.py --custom-allocator
