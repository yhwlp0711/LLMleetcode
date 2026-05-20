# PyTorch 张量操作热身

覆盖 PyTorch 张量常用操作。除非题目明确允许，**禁止写 Python `for`/`while`
循环**。每个函数独立判分。

## 待实现函数

### 1. `flatten_and_concat(a, b)`
给定形状任意的两个张量 `a` 和 `b`，返回一个一维张量：把 `a` 展平后拼接上
`b` 展平后的结果。

### 2. `row_softmax(x)`
给定二维张量 `x`（shape `(N, D)`），返回按行（最后一维）做 softmax 后的结
果。**必须用数值稳定写法**（先减去每行的最大值再 exp）。

### 3. `pairwise_squared_distance(x, y)`
给定 `x`（shape `(N, D)`）和 `y`（shape `(M, D)`），返回 shape `(N, M)`
的张量，其中 `out[i, j] = ||x[i] - y[j]||²`。**不许写 Python 循环**，用广
播或 `(a-b)² = a² - 2ab + b²` 技巧。

### 4. `masked_mean(x, mask)`
给定 `x`（shape `(B, T, D)`）和布尔 `mask`（shape `(B, T)`，True 表示有效），
返回 shape `(B, D)`，每个 batch 内有效位置的均值。假设每个 batch 至少有一
个有效位置。

### 5. `top_k_indices(scores, k)`
给定 `scores`（shape `(N,)`），返回 top-k 值的索引（一维张量，长度 `k`），
**按分数降序排列**。并列时索引较小者优先。

## 说明

- 所有输入都是 CPU 上的 `torch.float32` 张量。除非特别说明，返回值 dtype
  保持一致（索引必须是 `int64`）。
- 不要依赖 layout 技巧；判分时会用 contiguous 和 non-contiguous 两种输入。
