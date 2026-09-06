import os
import torch

# 1. Register the pluggable allocator via environment variables
new_alloc = torch.cuda.memory.CUDAPluggableAllocator('custom_allocator.so', 'my_malloc', 'my_free')
print("Testing PyTorch with custom MemoryManager...")

# PyTorch should intercept this and call  my_malloc
x = torch.ones((1024, 1024), device='cuda')

# Verifying correctness
print(f"Tensor allocated! Sum of elements: {x.sum().item()}")

# Trigger my_free
del x
torch.cuda.empty_cache()