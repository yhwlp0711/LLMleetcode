# 解题思路：手撕 matmul / 转置 / batched matmul

## 1. `matmul(A, B)`

矩阵乘的定义：`out[i, j] = sum_k A[i, k] * B[k, j]`。

最直接的写法是三重 `for`，但慢得离谱。利用题目允许的"输出行循环"，把内
层的两个循环（`k` 和 `j`）一起向量化：

```python
def matmul(A, B):
    M, K = A.shape
    _, N = B.shape
    out = np.empty((M, N), dtype=np.result_type(A, B))
    for i in range(M):
        # A[i, :, None] -> (K, 1), B -> (K, N)，相乘广播到 (K, N)，沿 K 求和
        out[i, :] = (A[i, :, None] * B).sum(axis=0)
    return out
```

**关键技巧**：`A[i, :, None]` 给行向量加一个新轴，shape 变 `(K, 1)`；与
`B` 的 `(K, N)` 广播相乘得到 `(K, N)`；`.sum(axis=0)` 沿 K 求和得 `(N,)`。

`np.result_type(A, B)` 自动推断输出 dtype（比如 float32 × float64 → float64）。

## 2. `transpose_2d(A)`

最易读的写法：用「行/列」两组索引广播，做花式索引赋值：

```python
def transpose_2d(A):
    M, N = A.shape
    out = np.empty((N, M), dtype=A.dtype)
    rows = np.arange(M)[:, None]   # (M, 1)
    cols = np.arange(N)[None, :]   # (1, N)
    out[cols, rows] = A[rows, cols]
    return out
```

`A[rows, cols]` 通过广播变成 `(M, N)`，每个元素是 `A[i, j]`。同理
`out[cols, rows]` 的 LHS 也是 `(M, N)`，但索引交换了，所以赋值后
`out[j, i] = A[i, j]`，正好是转置。

**进阶**：也可以直接 stride trick —— 转置只是 `(strides[1], strides[0])`，
不动数据本身：

```python
def transpose_2d(A):
    M, N = A.shape
    return np.lib.stride_tricks.as_strided(
        A, shape=(N, M), strides=(A.strides[1], A.strides[0]),
    )
```

但 `as_strided` 不安全（容易越界），日常代码尽量别用。

## 3. `batched_matmul(A, B)`

`np.einsum` 一行解决：

```python
def batched_matmul(A, B):
    return np.einsum("bmk,bkn->bmn", A, B)
```

`einsum` 字符串「bmk, bkn -> bmn」描述了：
- 输入 A 的轴叫 `b, m, k`
- 输入 B 的轴叫 `b, k, n`
- 输出的轴叫 `b, m, n`
- **重复出现但不在输出中的轴**（`k`）会被求和（contract）
- 在输入与输出都出现的轴（`b`）被保留

这就是 batched matmul：对每个 batch 单独做一次 `(M, K) @ (K, N)`。

## 性能补充

`np.einsum` 表达力强但有时不是最快的（依赖 BLAS）。生产中应直接用
`A @ B` / `np.matmul(A, B)`，它们会调底层的优化矩阵乘（如 OpenBLAS / MKL），
比手写循环快几个数量级。本题禁用只是为了练手。
