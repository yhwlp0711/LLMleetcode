# 解题思路：滑动窗口

## 一句话思路

三道子题的核心就是一个工具：**`sliding_window_view`**——它用 stride trick
零拷贝地把一维数组变成「窗口堆叠」的二维视图。有了这个，移动平均和
一维互相关（cross-correlation）都变成矩阵运算的一行代码。

## 拆解思路

### sliding_window_1d：窗口视图 + stride 切片

`np.lib.stride_tricks.sliding_window_view(x, window_shape=W)` 返回 shape
`(L - W + 1, W)` 的二维数组，每一行是 `x` 上一个连续窗口。底层不复制数据，
只是调整 strides 让索引「看到」重叠区域——所以是零拷贝、非常快。

默认 stride=1。要支持任意 stride，只需在行维度做等间隔切片 `[::stride]`。
最后 `np.ascontiguousarray` 保证返回的内存连续（避免下游需要 contiguous
layout 时出错）。

### moving_average：窗口均值

有了 `sliding_window_1d`，移动平均就是对每行求均值：`.mean(axis=1)`。

### conv1d_valid：互相关 = 窗口矩阵 × kernel 向量

互相关（cross-correlation）的定义：$\text{out}[i] = \sum_{j=0}^{K-1}
x[i+j] \cdot \text{kernel}[j]$。

把所有窗口堆成 `(L-K+1, K)` 的矩阵，每行与 kernel 做内积——这正好是
矩阵乘向量 `w @ kernel`，一行搞定。

注意：题目要互相关（不翻转 kernel），而数学卷积会把 kernel 翻转。深度学习
里的「卷积层」实际上做的也是互相关，所以这里和 PyTorch / TensorFlow 保持
一致。

## 参考实现

```python
import numpy as np

def sliding_window_1d(x: np.ndarray, window: int, stride: int) -> np.ndarray:
    full = np.lib.stride_tricks.sliding_window_view(x, window_shape=window)
    return np.ascontiguousarray(full[::stride])

def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    w = sliding_window_1d(x, window=window, stride=1)
    return w.mean(axis=1)

def conv1d_valid(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    K = kernel.shape[0]
    w = sliding_window_1d(x, window=K, stride=1)  # (L-K+1, K)
    return w @ kernel                              # (L-K+1,)
```

## 关键点

1. **`sliding_window_view` 是零拷贝的 view**：返回值和原数组共享内存。
   优势是快、省内存；代价是返回的数组是只读的（写入会影响原数组），且不一定
   是 C-contiguous 的。所以需要写入或传给要求连续内存的下游时，先
   `np.ascontiguousarray` 拷贝一份。

2. **互相关 vs 卷积**：数学卷积是 `sum(x[i:i+K] * kernel[::-1])`（kernel
   翻转），互相关是 `sum(x[i:i+K] * kernel)`（不翻转）。`np.convolve` 做的
   是数学卷积，所以如果用它实现本题要先 `kernel[::-1]`，反而绕一圈。
   直接 `w @ kernel` 最简洁。

3. **复杂度**：`sliding_window_view` 本身 O(1)（只算 strides），`.mean()` 和
   `@ kernel` 都是 O(输出长度 × 窗口大小)。如果窗口很大，可以用 `cumsum`
   前缀和差分把移动平均做到 O(L)；本题窗口一般不大，简单写法够用。

4. **延伸**：二维版本 `sliding_window_view(img, (H, W))` 可以生成图像上的
   patch 矩阵，用于实现 2D 卷积（im2col 算法）。这是从一维窗口到高维的
   自然推广。
