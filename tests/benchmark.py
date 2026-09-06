import os
import argparse
import time
import threading
import subprocess
import torch
import torch.nn as nn
import cupy as cp

class MemoryMonitor:
    def __init__(self):
        self.keep_measuring = True
        self.peak_memory_mb = 0

    def measure_memory(self):
        while self.keep_measuring:
            try:
                # Query nvidia-smi for VRAM usage
                result = subprocess.check_output(
                    ['nvidia-smi', '--query-compute-apps=used_memory', '--format=csv,nounits,noheader'],
                    encoding='utf-8'
                )
                if result.strip():
                    mem = sum(int(x) for x in result.strip().split('\n'))
                    if mem > self.peak_memory_mb:
                        self.peak_memory_mb = mem
            except Exception:
                pass
            time.sleep(0.1)

class CityLocationPredictor(nn.Module):
    def __init__(self, input_features, num_cities):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_cities)
        )

    def forward(self, x):
        return self.network(x)

def get_unified_cupy_allocator():
    def allocator(size):
        if size == 0:
            return cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(0, 0, None, 0), 0)
        tensor = torch.empty(size, dtype=torch.uint8, device='cuda')
        return cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(tensor.data_ptr(), size, tensor), 0)
    return allocator

def run_benchmark(use_custom_allocator):
    print(f"--- Starting Benchmark ---")
    print(f"Allocator: {'Custom Unified' if use_custom_allocator else 'Default Siloed'}")

    monitor = MemoryMonitor()
    monitor_thread = threading.Thread(target=monitor.measure_memory)
    monitor_thread.start()

    try:
        if use_custom_allocator:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            so_path = os.path.join(current_dir, '..', 'custom_allocator.so')

            # 1. Inject your custom manager into PyTorch
            new_alloc = torch.cuda.memory.CUDAPluggableAllocator(so_path, 'my_malloc', 'my_free')
            torch.cuda.memory.change_current_allocator(new_alloc)

            # 2. Force CuPy to use the injected PyTorch allocator
            cp.cuda.set_allocator(get_unified_cupy_allocator())

        print("Generating mock survey dataset in CuPy...")
        # Tune this up or down based on your GPU. 20M should stress most consumer GPUs.
        num_rows = 20_000_000
        num_features = 20
        num_cities = 50

        # PHASE 1: Allocate raw data (takes up heavy VRAM)
        raw_features = cp.random.randn(num_rows, num_features, dtype=cp.float32)

        print("Simulating data preprocessing (Fragmenting the pools)...")
        # PHASE 2: Process data (takes up MORE heavy VRAM)
        features_cp = (raw_features - cp.mean(raw_features, axis=0)) / cp.std(raw_features, axis=0)
        labels_cp = cp.random.randint(0, num_cities, size=num_rows, dtype=cp.int64)

        # PHASE 3: Delete raw data.
        # IN SILOED MODE: CuPy holds this memory in its internal pool. PyTorch cannot access it.
        # IN UNIFIED MODE: CuPy destroys the tensor. PyTorch immediately calls your C++ `my_free`.
        del raw_features

        print("Converting CuPy arrays to PyTorch tensors...")
        features_tensor = torch.as_tensor(features_cp, device='cuda')
        labels_tensor = torch.as_tensor(labels_cp, device='cuda')

        print("Initializing City Location model and training loop...")
        model = CityLocationPredictor(num_features, num_cities).cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        batch_size = 500_000
        epochs = 3

        start_time = time.time()
        for epoch in range(epochs):
            for i in range(0, num_rows, batch_size):
                batch_features = features_tensor[i:i+batch_size]
                batch_labels = labels_tensor[i:i+batch_size]

                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

            print(f"  Epoch {epoch+1}/{epochs} Complete. Loss: {loss.item():.4f}")

        print(f"Training completed in {time.time() - start_time:.2f} seconds.")

    except RuntimeError as e:
        print(f"\n[CRASH] Caught Runtime Error (Likely OOM): {e}")
    finally:
        monitor.keep_measuring = False
        monitor_thread.join()
        print(f"\n--- Benchmark Results ---")
        print(f"Peak VRAM Usage: {monitor.peak_memory_mb} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--custom-allocator', action='store_true', help='Use the custom unified allocator')
    args = parser.parse_args()

    run_benchmark(args.custom_allocator)