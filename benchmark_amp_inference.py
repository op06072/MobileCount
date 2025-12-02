import time
import torch
import torch.nn as nn
from config import cfg
from models.CC import CrowdCounter


def benchmark():
    device = cfg.DEVICE
    print(f"Benchmarking on device: {device}")

    # Initialize model
    model = CrowdCounter(cfg.GPU_ID, cfg.NET).to(device)
    model.eval()

    # Dummy input
    batch_size = 1
    c, h, w = 3, 768, 1024
    input_tensor = torch.randn(batch_size, c, h, w).to(device)

    # Warmup
    print("Warming up...")
    for _ in range(10):
        with torch.no_grad():
            _ = model(input_tensor)

    # Benchmark FP32
    print("Benchmarking FP32...")
    start_time = time.time()
    num_iters = 50
    with torch.no_grad():
        for _ in range(num_iters):
            _ = model(input_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    fp32_time = (time.time() - start_time) / num_iters
    print(f"FP32 Average Time: {fp32_time * 1000:.2f} ms")

    # Determine AMP dtype
    amp_dtype = torch.float16
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        print("Using bfloat16 for AMP.")
    else:
        print("Using float16 for AMP.")

    # Benchmark AMP
    print("Benchmarking AMP...")
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iters):
            with torch.amp.autocast(device.type, dtype=amp_dtype):
                _ = model(input_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    amp_time = (time.time() - start_time) / num_iters
    print(f"AMP Average Time: {amp_time * 1000:.2f} ms")

    speedup = fp32_time / amp_time
    print(f"Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    benchmark()
