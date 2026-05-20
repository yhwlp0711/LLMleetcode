"""Statistical checks for init / random tensors.

Useful for problems where the student must initialize parameters with a specific
scheme (e.g. Xavier/Kaiming) — we cannot compare exact values, but we can check
that the distribution statistics fall within expected bounds.
"""
from __future__ import annotations

from math import sqrt

import torch

from .compare import CompareResult


def check_shape(tensor: torch.Tensor, expected_shape: tuple[int, ...], *, name: str = "tensor") -> CompareResult:
    if tuple(tensor.shape) != tuple(expected_shape):
        return CompareResult(
            passed=False,
            reason=f"{name} shape mismatch: expected {expected_shape}, got {tuple(tensor.shape)}",
        )
    return CompareResult(passed=True)


def check_init_stats(
    tensor: torch.Tensor,
    *,
    expected_mean: float = 0.0,
    expected_std: float | None = None,
    mean_tol: float = 0.1,
    std_rel_tol: float = 0.2,
    name: str = "tensor",
) -> CompareResult:
    """Check that a randomly initialized tensor has the expected mean/std.

    Tolerances are relatively loose by default since finite-sample mean/std
    fluctuate. Tighten for very large tensors.
    """
    t = tensor.detach().to(torch.float64).reshape(-1)
    if t.numel() < 2:
        return CompareResult(passed=False, reason=f"{name} too small to check stats")
    m = float(t.mean().item())
    s = float(t.std(unbiased=True).item())

    if abs(m - expected_mean) > mean_tol:
        return CompareResult(
            passed=False,
            reason=f"{name} mean {m:.4f} not within {mean_tol} of expected {expected_mean:.4f}",
            actual_preview=f"mean={m:.4f}, std={s:.4f}",
            expected_preview=f"mean≈{expected_mean:.4f}, std≈{expected_std}",
        )
    if expected_std is not None:
        if abs(s - expected_std) > std_rel_tol * expected_std:
            return CompareResult(
                passed=False,
                reason=(
                    f"{name} std {s:.4f} not within {std_rel_tol*100:.0f}% of "
                    f"expected {expected_std:.4f}"
                ),
                actual_preview=f"mean={m:.4f}, std={s:.4f}",
                expected_preview=f"mean≈{expected_mean:.4f}, std≈{expected_std:.4f}",
            )
    return CompareResult(
        passed=True,
        actual_preview=f"mean={m:.4f}, std={s:.4f}",
        expected_preview=f"mean≈{expected_mean:.4f}, std≈{expected_std}" if expected_std is not None else "",
    )


def xavier_uniform_std(fan_in: int, fan_out: int, gain: float = 1.0) -> float:
    """Theoretical std of Xavier-uniform init: U(-a, a) with a = gain * sqrt(6/(fan_in+fan_out))."""
    a = gain * sqrt(6.0 / (fan_in + fan_out))
    return a / sqrt(3.0)


def kaiming_normal_std(fan_in: int, gain: float = sqrt(2.0)) -> float:
    """Theoretical std of Kaiming-normal init: N(0, gain/sqrt(fan_in))."""
    return gain / sqrt(fan_in)
