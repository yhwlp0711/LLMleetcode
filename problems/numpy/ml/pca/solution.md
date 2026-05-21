# 解题思路：PCA

## 核心定理

PCA 的最优解 = 数据中心化后协方差矩阵的**前 k 个特征向量**。

证明可以从两个角度（最大方差 / 最小重构误差）出发，都得到同一个解。

## 为什么用 SVD 而不是直接特征分解？

数学上等价：
- 协方差矩阵 $C = \frac{1}{N-1} X\_c^\top X\_c$（$D \times D$）
- 特征分解 $C = V \Lambda V^\top$

但**数值上 SVD 更稳定**：
- 不需要显式构造 $X^\top X$（避免条件数平方）
- 一步到位拿到 U/S/V

代码：

```python
U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
# X_c = U @ diag(S) @ Vt
# Vt 的每一行就是主成分方向
# S^2 / (N-1) 就是特征值（方差）
```

## 参考实现

```python
def pca(X, n_components):
    N = X.shape[0]
    X_c = X - X.mean(axis=0, keepdims=True)              # 中心化

    U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    components = Vt[:n_components].copy()                # (k, D)
    explained_var = (S[:n_components] ** 2) / (N - 1)    # (k,)
    projected = X_c @ components.T                       # (N, k)

    _fix_sign(components, projected)
    return components, explained_var, projected
```

## 三个关键点

### 1. 用 `full_matrices=False`

不加这个，SVD 会算完整的 `U: (N, N)` 和 `Vt: (D, D)`，对 N >> D 时巨慢
且浪费内存。`full_matrices=False` 只算最小必需的 `(N, min(N,D))` 和
`(min(N,D), D)`。

### 2. 方差用 `S² / (N-1)`

样本协方差是 `X^T X / (N-1)`（无偏估计，`ddof=1`）。所以特征值是 `S² /
(N-1)`。**容易写错为 `S² / N`**，差一个 N/(N-1) 的因子，在 N 小时差距明
显。

### 3. 符号歧义 —— **必须统一**

SVD 的 V 和 -V 都是合法解（特征向量符号不唯一）。不同 LAPACK 实现可能
给不同符号；用户和参考实现的环境只要一致就行，但**这是无法保证的**。

判分必须强制一个约定。本题约定：**每个主成分第一个非零元素为正**。
这是 sklearn 内部用的标准化方法，叫 "svd_flip"。

```python
def _fix_sign(components, projected):
    for i in range(components.shape[0]):
        nonzero = np.abs(components[i]) > 1e-12
        if not nonzero.any():
            continue
        first = np.argmax(nonzero)
        if components[i, first] < 0:
            components[i] = -components[i]
            projected[:, i] = -projected[:, i]
```

**注意 `projected` 也要同步翻号** —— 它是 `X_c @ components.T`，components
翻号后 projected 对应列也翻号才一致。

## 易错点

### 1. 忘了中心化

PCA 公式假设数据均值为 0。如果不中心化，第一个主成分会朝向数据的均值方
向，完全错了。

### 2. 主成分按行还是按列？

约定：`components` 是 `(k, D)`，**每行一个主方向**（跟 sklearn 一致）。
也有些文献写成 `(D, k)`（每列一个）。本题按 sklearn 约定。

那 projection 就是 `X_c @ components.T`（注意 `.T`），结果 `(N, k)`。

### 3. `explained_var` 排好序

SVD 返回的 S 已经是降序，所以前 k 个就是最大的 k 个方差。**不要手动再排
序**，否则可能搞反。

## 进阶：奇异值的工业用法

PCA 还有几个常见衍生输出：

- **explained_variance_ratio**: `explained_var / explained_var.sum()` ——
  每个主成分占总方差的比例
- **cumulative explained variance**: `np.cumsum(ratio)` —— 用来挑选 k
  （比如保留 95% 方差需要多少维）
- **whitening**: `projected / np.sqrt(explained_var)` —— 让各方向方差为
  1，常用于 ZCA 等
