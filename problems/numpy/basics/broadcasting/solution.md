# 解题思路：广播与外积运算

## 一句话思路

这题练的是广播（broadcasting）的核心招式：**用 `None`（即 `np.newaxis`）
给数组插入长度为 1 的轴**，让 NumPy 自动把维度对齐后做逐元素运算，从而
避免所有 Python 循环。

## 拆解思路

### 广播规则速记

NumPy 广播的三条规则：

1. 从**最后一维**开始对齐。
2. 两个维度相等，或其中一个是 1 → 可以广播。
3. 缺失的维度视为 1。

关键操作：`a[:, None]` 把 shape `(N,)` 变成 `(N, 1)`，`b[None, :]` 把
shape `(M,)` 变成 `(1, M)`。两者做算术运算时广播成 `(N, M)`——这就是
「外积模式」。

### 四个子函数的思路

**outer_sum**：`a` 变列向量 `(N, 1)`，`b` 变行向量 `(1, M)`，相加广播成
`(N, M)`，即 `R[i, j] = a[i] + b[j]`。

**pairwise_difference**：和 outer_sum 同模式，`a = b = x`，用减法：
`x[:, None] - x[None, :]`，得到 `(N, N)` 的反对称矩阵。

**normalize_columns**：沿 `axis=0`（列方向）求均值和标准差，得到 shape
`(1, D)`（加了 `keepdims=True`），再用原数组 `(N, D)` 减 / 除，广播自动
在行方向展开。

**apply_per_row_scale**：`s` 是 `(N,)`，想按行缩放就要变成 `(N, 1)`，
这样 `(N, 1) * (N, D)` 广播成 `(N, D)`——每行乘各自的标量。

## 参考实现

```python
import numpy as np

def outer_sum(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[:, None] + b[None, :]

def pairwise_difference(x: np.ndarray) -> np.ndarray:
    return x[:, None] - x[None, :]

def normalize_columns(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=0, keepdims=True)          # (1, D)
    std = X.std(axis=0, ddof=0, keepdims=True)    # (1, D)
    return (X - mean) / std

def apply_per_row_scale(X: np.ndarray, s: np.ndarray) -> np.ndarray:
    return s[:, None] * X
```

## 关键点

1. **`[:, None]` 是最常用的升维方式**：等价于 `np.expand_dims(a, axis=1)`
   或 `a.reshape(-1, 1)`，但更简洁。记住「想让哪个维度变成 1，就在对应位置
   插 `None`」。

2. **`keepdims=True` 让归约（reduce）后维度不消失**：`mean(axis=0)` 默认
   输出 shape `(D,)`，加 `keepdims=True` 变成 `(1, D)`，这样和 `(N, D)` 做
   算术时广播方向明确——沿 axis=0 展开。不加 `keepdims` 在本例也行（尾维度
   对齐），但显式保留更不容易出错。

3. **`ddof=0` vs `ddof=1`**：`std(ddof=0)` 除以 $N$（总体标准差），
   `ddof=1` 除以 $N-1$（样本标准差）。本题要求总体标准差，写错 ddof 数值
   会偏。

4. **延伸**：外积模式 `a[:, None] ○ b[None, :]` 把 `+` 换成 `*` 就是
   `np.outer(a, b)`。KNN 中算成对距离矩阵（见 `numpy.ml.knn`）也是同一招
   的变体。
