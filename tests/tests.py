import os
import torch

# Resolves the path to the parent directory where the .so was likely compiled
current_dir = os.path.dirname(os.path.abspath(__file__))
so_path = os.path.join(current_dir, '..', 'custom_allocator.so')


# Register the pluggable allocator via environment variables
new_alloc = torch.cuda.memory.CUDAPluggableAllocator(so_path, 'my_malloc', 'my_free')
torch.cuda.memory.change_current_allocator(new_alloc)
print("Testing PyTorch with custom MemoryManager...")

# PyTorch should intercept this and call  my_malloc
x = torch.ones((1024, 1024), device='cuda')

# Verifying correctness
print(f"Tensor allocated! Sum of elements: {x.sum().item()}")

# Trigger my_free
del x
torch.cuda.empty_cache()