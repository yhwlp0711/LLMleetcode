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
    """GPU fp32 accumulation order differs from CPU, so relax overly tight
    tolerances a bit on non-CPU devices to avoid false negatives on otherwise
    correct submissions."""
    dev_type = getattr(device, "type", None) if device is not None else None
    if dev_type in ("mps", "cuda"):
        return max(atol, 1e-5), max(rtol, 1e-4)
    return atol, rtol
