# PCA 主成分分析

实现 PCA 的拟合与投影：找数据中方差最大的方向，把数据投影到低维。

## 函数签名

```python
def pca(
    X: np.ndarray,    # (N, D) 数据
    n_components: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    返回:
        components:    (n_components, D)   前 k 个主方向，按方差从大到小排
        explained_var: (n_components,)     对应的方差（特征值），降序
        projected:     (N, n_components)   X 在主成分上的投影
    """
```

## 算法

1. **中心化**：`X_c = X - X.mean(axis=0)`
2. **SVD 分解**：`U, S, Vt = svd(X_c, full_matrices=False)`
3. **主成分**：前 k 行的 `Vt`（每行是一个主方向）
4. **方差**：`S² / (N - 1)`（样本协方差对应的特征值，**用 ddof=1**）
5. **投影**：`X_c @ components.T`

## 符号约定

主成分的方向可以差一个符号（`v` 和 `-v` 张成同一个子空间，特征向量符号
不唯一）。为了判分一致：

**强制每个主成分的第一个非零元素为正**。如果不是，把这个主成分整体翻
号（同时翻转 projected 对应列）。

具体做法：
```python
for i in range(n_components):
    # 找该主成分中第一个 abs > eps 的元素
    first_nonzero = ...
    if components[i, first_nonzero] < 0:
        components[i] *= -1
        projected[:, i] *= -1
```

## 说明

- 输入 `np.float64`。
- **禁用** `sklearn`。**允许用** `np.linalg.svd` / `np.linalg.eigh`。
- 容差 `atol=1e-8`（双精度 SVD 应该很准）。
- 假设 `n_components <= min(N-1, D)`。
