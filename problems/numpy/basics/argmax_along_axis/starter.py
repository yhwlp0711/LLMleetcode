"""按轴 argmax 实现。"""

from __future__ import annotations

import numpy as np


def argmax_along_axis(x: np.ndarray, axis: int) -> np.ndarray:
    # TODO: 返回 x 在指定 axis 上最大值的索引。
    # 限制：禁用 np.argmax / np.argpartition。
    # 提示：先把 axis 搬到最后，然后用 np.where 沿着最后一维做扫描更新。
    raise NotImplementedError
