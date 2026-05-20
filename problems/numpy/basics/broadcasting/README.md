# 广播与外积运算

练习 NumPy 的 **broadcasting**，实现几个常见的外积式操作 —— **全程不允许写
Python 循环**。

## 待实现函数

### 1. `outer_sum(a, b)`
给定一维数组 `a`（shape `(N,)`）和 `b`（shape `(M,)`），返回二维数组 `R`
（shape `(N, M)`），其中 `R[i, j] = a[i] + b[j]`。

### 2. `pairwise_difference(x)`
给定一维数组 `x`（shape `(N,)`），返回二维数组 `D`（shape `(N, N)`），
其中 `D[i, j] = x[i] - x[j]`。

### 3. `normalize_columns(X)`
给定二维数组 `X`（shape `(N, D)`），返回相同 shape 的 `X_norm`，每**列**做零
均值、单位方差归一化：`X_norm[:, j] = (X[:, j] - mean_j) / std_j`。使用总
体方差（`ddof=0`）。假设没有恒值列。

### 4. `apply_per_row_scale(X, s)`
给定 `X`（shape `(N, D)`）和 `s`（shape `(N,)`），把 `X` 的每一行乘以对应
的标量 `s[i]`，即 `out[i] = s[i] * X[i]`。

## 要求

- 输入均为 `np.float64`，输出保持同 dtype。
- 任何地方都不要写 `for` / `while`。
