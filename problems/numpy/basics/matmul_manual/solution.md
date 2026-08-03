# 解题思路：手撕 matmul / 转置 / batched matmul

## 一句话思路

三道子题考的是线性代数原语的**底层理解**：矩阵乘就是「对输出行循环 +
广播消去内维」；转置就是「行列索引互换」；batched matmul 用 `einsum`
一行搞定。核心难点在于用广播（broadcasting）把内层循环向量化。

## 拆解思路

### matmul：行循环 + 广播求和

矩阵乘定义：$C_{ij} = \sum_k A_{ik} B_{kj}$。

三重循环太慢，但题目允许「对输出行循环」。关键观察：固定第 $i$ 行，
$C[i, :] = \sum_k A[i, k] \cdot B[k, :]$。如果把 $A[i, :]$ 升维成
`(K, 1)`，与 $B$ 的 `(K, N)` 广播相乘得 `(K, N)`，再沿 axis=0 求和得
`(N,)` —— 一行搞定一整行输出。

### transpose：索引互换

转置的本质是 `out[j, i] = A[i, j]`。构造两组索引 `rows = arange(M)[:,
None]` 和 `cols = arange(N)[None, :]`，利用广播让它们组成所有 `(i, j)` 对，
然后做花式索引（fancy indexing）赋值 `out[cols, rows] = A[rows, cols]`。

### batched matmul：einsum 下标描述

`np.einsum("bmk,bkn->bmn", A, B)` 描述了：
- 输入 A 的轴叫 `b, m, k`；输入 B 的轴叫 `b, k, n`
- 输出保留 `b, m, n`；`k` 出现在输入但不在输出 → 被求和（contract）

这就是对每个 batch 做一次 `(M, K) @ (K, N)`。

## 参考实现

```python
import numpy as np

def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    M, K = A.shape
    K2, N = B.shape
    out = np.empty((M, N), dtype=np.result_type(A, B))
    for i in range(M):
        out[i, :] = (A[i, :, None] * B).sum(axis=0)  # (K,1)*(K,N)->(K,N)->sum->(N,)
    return out

def transpose_2d(A: np.ndarray) -> np.ndarray:
    M, N = A.shape
    out = np.empty((N, M), dtype=A.dtype)
    rows = np.arange(M)[:, None]   # (M, 1)
    cols = np.arange(N)[None, :]   # (1, N)
    out[cols, rows] = A[rows, cols]
    return out

def batched_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.einsum("bmk,bkn->bmn", A, B)
```

## 关键点

1. **`A[i, :, None]` 的升维技巧**：把形状 `(K,)` 变成 `(K, 1)`，与 `B` 的
   `(K, N)` 广播相乘得 `(K, N)`，再 `.sum(axis=0)` 得 `(N,)`。这把矩阵乘
   的「内积」变成了一次广播乘 + 归约（reduce），避免了内层循环。

2. **`np.result_type(A, B)` 自动推断输出 dtype**：比如 float32 × float64 →
   float64。显式指定 dtype 是好习惯，避免默认 float64 带来不必要的精度提升
   或丢失。

3. **花式索引赋值实现转置**：`out[cols, rows] = A[rows, cols]` 中，
   `rows` 和 `cols` 通过广播展开成 `(M, N)` 的索引网格，赋值时行列互换就
   完成了转置。这比双重循环优雅且快得多。

4. **延伸**：生产代码直接用 `@` / `np.matmul`，它们调用底层 BLAS 库
   （OpenBLAS / MKL），比手写循环快几个数量级。本题禁用只是为了从 index
   层面理解矩阵乘的本质。
