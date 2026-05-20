"""Device selection: prefer CUDA, then MPS, then CPU."""
from __future__ import annotations

import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_tolerance(device, atol: float, rtol: float) -> tuple[float, float]:
    """MPS fp32 precision is slightly lower than CUDA/CPU; relax tolerance a bit."""
    if device is not None and getattr(device, "type", None) == "mps":
        return max(atol, 1e-5), max(rtol, 1e-4)
    return atol, rtol
