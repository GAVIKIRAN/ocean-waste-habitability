"""Verify that PyTorch sees the CUDA GPU and can run FP16 maths on it.

Usage::

    python scripts/check_gpu.py
Exit code 0 = CUDA OK, 1 = CPU only (the project still runs, just slower).
"""
from __future__ import annotations

import sys
import time


def main() -> int:
    import torch

    print(f"PyTorch        : {torch.__version__}")
    print(f"Built for CUDA : {torch.version.cuda}")
    print(f"cuDNN          : {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'n/a'}")
    if not torch.cuda.is_available():
        print("CUDA available : NO -> training/inference will run on CPU.")
        print("Fix: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 (see README)")
        return 1
    dev = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(dev)
    print(f"CUDA available : yes ({torch.cuda.device_count()} device(s))")
    print(f"GPU 0          : {props.name}  |  {props.total_memory / 1024**3:.1f} GB VRAM  |  compute {props.major}.{props.minor}")
    a = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
    b = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
    for _ in range(3):  # warm-up (kernel selection, clocks)
        a @ b
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(10):
        c = a @ b
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / 10
    print(f"FP16 matmul    : {2 * 4096**3 / dt / 1e12:.1f} TFLOPS  (result finite: {bool(torch.isfinite(c).all())})")
    free, total = torch.cuda.mem_get_info()
    print(f"VRAM free      : {free / 1024**3:.1f} / {total / 1024**3:.1f} GB")
    print(f"Training device: 0 (device = 0 if torch.cuda.is_available() else 'cpu')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
