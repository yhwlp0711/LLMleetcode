"""Numerical comparison utilities for judging."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class CompareResult:
    passed: bool
    reason: str = ""
    max_abs_diff: float | None = None
    max_rel_diff: float | None = None
    expected_preview: str = ""
    actual_preview: str = ""


def _to_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    if isinstance(x, (int, float, list, tuple)):
        return torch.tensor(x)
    raise TypeError(f"Unsupported type for comparison: {type(x).__name__}")


def _preview(t: torch.Tensor, n: int = 6) -> str:
    if t.ndim == 0:
        return f"shape=(), dtype={t.dtype}, value={t.item():.6g}"
    flat = t.reshape(-1)
    head = flat[:n].tolist()
    suffix = " ..." if flat.numel() > n else ""
    formatted = ", ".join(f"{v:.6g}" for v in head)
    return f"shape={tuple(t.shape)}, dtype={t.dtype}, [{formatted}{suffix}]"


def compare_numeric(
    actual: Any,
    expected: Any,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> CompareResult:
    """Compare two tensor-like objects with allclose semantics."""
    try:
        a = _to_tensor(actual)
        e = _to_tensor(expected)
    except TypeError as exc:
        return CompareResult(passed=False, reason=str(exc))

    if a.shape != e.shape:
        return CompareResult(
            passed=False,
            reason=f"shape mismatch: expected {tuple(e.shape)}, got {tuple(a.shape)}",
            expected_preview=_preview(e),
            actual_preview=_preview(a),
        )

    a_f = a.to(torch.float64)
    e_f = e.to(torch.float64)
    abs_diff = (a_f - e_f).abs()
    max_abs = float(abs_diff.max().item()) if abs_diff.numel() else 0.0
    denom = e_f.abs().clamp(min=1e-12)
    max_rel = float((abs_diff / denom).max().item()) if abs_diff.numel() else 0.0

    passed = torch.allclose(a_f, e_f, atol=atol, rtol=rtol)
    return CompareResult(
        passed=passed,
        reason="" if passed else f"values differ beyond tolerance (atol={atol}, rtol={rtol})",
        max_abs_diff=max_abs,
        max_rel_diff=max_rel,
        expected_preview=_preview(e),
        actual_preview=_preview(a),
    )
