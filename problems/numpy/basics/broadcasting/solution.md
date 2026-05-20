# 解题思路：广播与外积运算

## 核心知识点：广播规则

NumPy broadcasting 在两个 shape 维度对齐时按以下规则：

1. 从**最后一维**开始对齐。
2. 维度相等，或其中一个是 1，可以广播。
3. 缺失的维度视为 1。

构造广播的核心招式是 `None`（等价于 `np.newaxis`）：在指定位置插入一个长度
为 1 的轴。

---

## 1. `outer_sum(a, b)`

把 `a` 变成列向量 `(N, 1)`，`b` 变成行向量 `(1, M)`，相加即可广播成 `(N, M)`：

```python
def outer_sum(a, b):
    return a[:, None] + b[None, :]
```

这是经典的「外积模式」。把 `+` 换成 `*` 就是 `np.outer`。

---

## 2. `pairwise_difference(x)`

跟上面同模式，但 `a = b = x`：

```python
def pairwise_difference(x):
    return x[:, None] - x[None, :]
```

`D[i, j] = x[i] - x[j]`，对角线全 0，反对称矩阵。

---

## 3. `normalize_columns(X)`

`X.mean(axis=0)` 对**列**求均值，结果 shape 是 `(D,)`。直接用 `(N, D) - (D,)`
广播规则会沿第 0 轴自动扩展，**也可以**。但为了显式清楚，加 `keepdims=True`
让 shape 变成 `(1, D)`：

```python
def normalize_columns(X):
    mean = X.mean(axis=0, keepdims=True)             # (1, D)
    std = X.std(axis=0, ddof=0, keepdims=True)        # (1, D)
    return (X - mean) / std
```

注意 `ddof=0`（总体方差，除以 `N`）vs `ddof=1`（样本方差，除以 `N-1`），
这道题要求 `ddof=0`。

---

## 4. `apply_per_row_scale(X, s)`

`s` 是 `(N,)`，想按行缩放就要把它变成列向量 `(N, 1)`，这样 `(N, 1) * (N, D)`
广播成 `(N, D)`：

```python
def apply_per_row_scale(X, s):
    return s[:, None] * X
```

如果写成 `s * X`，会按最后一维对齐 → shape 不匹配（除非 `N == D`）。

---

## 参考代码

`solution.py` 见下方代码块（CLI 已直接输出）。
