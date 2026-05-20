# 解题思路：滑动窗口

## 关键工具：`np.lib.stride_tricks.sliding_window_view`

NumPy 1.20+ 提供的官方滑窗 API。底层用 stride tricks 实现，**零拷贝**，
速度极快：

```python
np.lib.stride_tricks.sliding_window_view(x, window_shape=W)
# x: (L,) -> (L - W + 1, W)
```

返回的是 `x` 的一个 view（共享内存），所以**只读语义**，别尝试就地修改。

## 1. `sliding_window_1d(x, window, stride)`

`sliding_window_view` 默认 stride=1，要支持任意 stride，只需在第 0 维做切片：

```python
def sliding_window_1d(x, window, stride):
    full = np.lib.stride_tricks.sliding_window_view(x, window_shape=window)
    return np.ascontiguousarray(full[::stride])
```

`np.ascontiguousarray` 保证返回值在内存里是连续的（避免下游需要 contiguous
时出错；判分时 `allclose` 不要求 contiguous，但养成习惯）。

## 2. `moving_average(x, window)`

直接复用 #1，然后按行求均值：

```python
def moving_average(x, window):
    w = sliding_window_1d(x, window=window, stride=1)
    return w.mean(axis=1)
```

> **进阶做法（O(L)）**：用 `np.cumsum` 做前缀和差分，避免 O(LW) 复杂度。对
> 大 window 很有用；这里 W 小，simple is best。

## 3. `conv1d_valid(x, kernel)`

「互相关」= 窗口与 kernel 逐元素乘后求和。窗口堆叠后正好是矩阵 × 向量：

```python
def conv1d_valid(x, kernel):
    K = kernel.shape[0]
    w = sliding_window_1d(x, window=K, stride=1)  # (L-K+1, K)
    return w @ kernel                              # (L-K+1,)
```

**易错点**：题目要求互相关（cross-correlation），不是数学卷积。数学卷积
需要先把 kernel 翻转：`out[i] = sum(x[i:i+K] * kernel[::-1])`。深度学习里
习惯叫「卷积」其实都是互相关，所以这里跟 PyTorch / TF 的 `conv` 保持一致，
不翻转。

## 为什么不直接用 `np.convolve`？

`np.convolve(x, kernel, mode='valid')` 会按数学卷积翻转 kernel；要等价于
本题就得先 `kernel[::-1]`，反而绕一圈。`sliding_window_view + @` 更直观也
更高效。
