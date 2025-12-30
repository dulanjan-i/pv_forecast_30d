import torch
import time
import os

# FORCE ISOLATION
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("SLURM_ID_PASS", "0")

print(f"--- HARDWARE CHECK ---")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Count: {torch.cuda.device_count()}")
else:
    print("NO GPU DETECTED. FAILURE.")
    exit(1)

# CREATE MASSIVE TENSORS (Simulate your TFT batch)
BATCH_SIZE = 4096
DIM = 512
print(f"\n--- SPEED TEST ---")
print(f"Matrix Size: {BATCH_SIZE} x {DIM} (Float32)")
print("Initializing tensors...")

a = torch.randn(BATCH_SIZE, DIM, device="cuda")
b = torch.randn(DIM, DIM, device="cuda")

# WARMUP
print("Warming up GPU...")
for _ in range(10):
    c = torch.matmul(a, b)
torch.cuda.synchronize()

# BENCHMARK
print("Running 1000 Matrix Multiplications...")
start = time.time()
for i in range(1000):
    c = torch.matmul(a, b)
    # Simulate a tiny bit of CPU work (like a training loop)
    if i % 100 == 0:
        print(f"Step {i}...")

torch.cuda.synchronize()
end = time.time()

total_time = end - start
its = 1000 / total_time

print(f"\n=======================")
print(f"FINAL SPEED: {its:.2f} it/s")
print(f"Total Time:  {total_time:.2f}s")
print(f"=======================")

if its > 100:
    print("VERDICT: HARDWARE IS PERFECT. The issue is your TFT code/libraries.")
else:
    print("VERDICT: HARDWARE IS BROKEN. The container or node is throttling.")