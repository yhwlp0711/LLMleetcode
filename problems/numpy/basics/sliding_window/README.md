# 滑动窗口

实现时序处理 / 卷积中常用的滑窗工具。

## 待实现函数

### 1. `sliding_window_1d(x, window, stride)`
给定一维数组 `x`（长度 `L`），返回二维数组（shape 为
`((L - window) // stride + 1, window)`），每一行是 `x` 上一个连续窗口。

示例：`x=[1,2,3,4,5,6], window=3, stride=2 → [[1,2,3], [3,4,5]]`

### 2. `moving_average(x, window)`
给定一维数组 `x`（长度 `L`），返回长度 `L - window + 1` 的一维数组，第 `i`
个元素是 `x[i : i+window]` 的均值。

### 3. `conv1d_valid(x, kernel)`
给定 `x`（shape `(L,)`）与 `kernel`（shape `(K,)`），返回一维**互相关**
（`valid` 模式），输出 shape `(L - K + 1,)`，其中
`out[i] = sum(x[i : i+K] * kernel)`。**注意是互相关（cross-correlation），
不是数学意义上的卷积**，不要把 kernel 翻转。

## 说明

- 输入均为 `np.float64`。
- 可以用 `np.lib.stride_tricks.sliding_window_view`，但不强制 —— 任何正确
  的向量化实现都行。
- 假设输入合法（`L >= window > 0`，`stride > 0`，`L >= K > 0`）。
