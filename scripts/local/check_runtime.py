"""Check the local CUDA stack by executing both Torch and Warp GPU kernels."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch
import warp as wp


@wp.kernel
def _increment(values: wp.array(dtype=wp.float32)):
    index = wp.tid()
    values[index] = values[index] + 1.0


def main() -> None:
    print(f"Platform: {platform.platform()}", flush=True)
    print(f"Python: {platform.python_version()}", flush=True)
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '(all)')}")
    memory = {
        key: int(value.split()[0])
        for key, value in (line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    }
    print(f"System RAM: total={memory['MemTotal'] / 2**20:.1f} GiB, available={memory['MemAvailable'] / 2**20:.1f} GiB")
    subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free", "--format=csv,noheader"],
        check=True,
    )
    for package in ("torch", "warp-lang", "mujoco", "mujoco-warp", "mjlab"):
        print(f"{package}: {importlib.metadata.version(package)}", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA; stop before training.")
    print(f"Torch CUDA runtime: {torch.version.cuda}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 2**30:.2f} GiB")
    torch.manual_seed(0)
    x = torch.randn(128, 128, device="cuda:0", requires_grad=True)
    (x @ x).square().mean().backward()
    torch.cuda.synchronize()
    if x.grad is None or not torch.isfinite(x.grad).all():
        raise RuntimeError("Torch CUDA forward/backward returned invalid gradients.")

    wp.init()
    device = wp.get_device("cuda:0")
    if not device.is_cuda:
        raise RuntimeError("Warp selected a CPU device.")
    values = wp.zeros(128, dtype=wp.float32, device=device)
    wp.launch(_increment, dim=128, inputs=[values], device=device)
    wp.synchronize_device(device)
    np.testing.assert_array_equal(values.numpy(), np.ones(128, dtype=np.float32))
    print("PASS: Torch forward/backward and a compiled Warp CUDA kernel.", flush=True)
    print("Next: run scripts/local/smoke_train.py to validate MuJoCo-Warp and the trainer.")


if __name__ == "__main__":
    main()
