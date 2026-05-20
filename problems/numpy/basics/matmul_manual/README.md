# 手撕 matmul / 转置 / batched matmul

实现核心线性代数原语，**不准用 `np.dot` / `np.matmul` / `@` 运算符 /
`np.transpose` / `.T`**。目的是从 index 层面理解这些操作。

## 待实现函数

### 1. `matmul(A, B)`
给定 `A`（shape `(M, K)`）与 `B`（shape `(K, N)`），返回矩阵乘积
（shape `(M, N)`）。允许**只在输出行**上写循环；内层「点积」必须用
向量化 sum + 元素乘。

**禁用**：`np.dot`、`np.matmul`、`@` 运算符、`np.einsum`。

### 2. `transpose_2d(A)`
返回二维数组的转置。**禁用** `np.transpose`、`.T`、`np.swapaxes`。可以用
`np.empty` + 高级索引，或者 `np.reshape` + `np.lib.stride_tricks`。

### 3. `batched_matmul(A, B)`
给定 `A`（shape `(B, M, K)`）与 `B`（shape `(B, K, N)`），返回 batched
matmul（shape `(B, M, N)`）。**这里可以用** `np.einsum`，目标是一行解决。

## 说明

- 输入均为 `np.float64`。
- 容差 `1e-10`。
