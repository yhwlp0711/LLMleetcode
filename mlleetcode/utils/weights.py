"""Helpers for transferring weights between user and reference nn.Module instances."""
from __future__ import annotations

from typing import Any

import torch


def sync_weights(target: "torch.nn.Module", source: "torch.nn.Module", *, strict: bool = True) -> None:
    """Copy parameters from `source` into `target` in-place.

    Used in module-style problems so that the user's forward is judged against
    the reference forward with identical weights, isolating algorithmic
    correctness from init-related differences.
    """
    target.load_state_dict(source.state_dict(), strict=strict)


def assert_param_names(module: "torch.nn.Module", expected: list[str]) -> tuple[bool, str]:
    """Verify the module exposes the expected set of named parameters."""
    have = set(name for name, _ in module.named_parameters())
    want = set(expected)
    missing = want - have
    extra = have - want
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing: {sorted(missing)}")
        if extra:
            parts.append(f"extra: {sorted(extra)}")
        return False, "; ".join(parts)
    return True, ""
