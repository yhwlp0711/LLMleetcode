"""Reference: Argmax Along Axis."""

from __future__ import annotations

import numpy as np


def argmax_along_axis(x: np.ndarray, axis: int) -> np.ndarray:
    # Move target axis to the end, then iterate via vectorized scan.
    moved = np.moveaxis(x, axis, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    N, D = flat.shape

    # Iterate over D positions; pick the index where a new maximum is found.
    # We use a vectorized form: for each row, track current_max and current_idx.
    best_idx = np.zeros(N, dtype=np.int64)
    best_val = flat[:, 0].copy()
    for j in range(1, D):
        col = flat[:, j]
        better = col > best_val  # strict >, so ties keep the earlier idx
        best_val = np.where(better, col, best_val)
        best_idx = np.where(better, j, best_idx)

    out_shape = moved.shape[:-1]
    return best_idx.reshape(out_shape)
