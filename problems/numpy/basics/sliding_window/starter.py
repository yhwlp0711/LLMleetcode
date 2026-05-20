"""滑动窗口工具 —— 三个函数。"""

from __future__ import annotations

import numpy as np


def sliding_window_1d(x: np.ndarray, window: int, stride: int) -> np.ndarray:
    # TODO: 返回二维数组，每行是 x 上一个连续窗口。
    raise NotImplementedError


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    # TODO: 每个窗口的均值。输出长度 = len(x) - window + 1。
    raise NotImplementedError


def conv1d_valid(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    # TODO: 一维互相关，valid 模式。不要翻转 kernel！
    raise NotImplementedError
