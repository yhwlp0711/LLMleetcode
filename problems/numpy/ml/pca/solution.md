# 解题思路：PCA 主成分分析

## 一句话思路

主成分分析（PCA, Principal Component Analysis）的目标是：找到数据中**方差
最大的几个方向**，把高维数据投影到低维。实现上就是「中心化 → SVD 分解 →
取前 k 行」三步，难点在于理解为什么 SVD 能给出主成分，以及符号歧义的处理。

## 从直觉到公式

### 直觉：找「数据最散开」的方向

假设数据在一个 D 维空间里。我们想找一个方向 $v$，使得数据投影到 $v$ 上后
「最分散」（方差最大）。写成优化问题：

$$\max_{\|v\|=1} \text{Var}(Xv) = \max_{\|v\|=1} v^\top C v$$

其中 $C$ 是样本协方差矩阵（covariance matrix）。这个问题的解就是 $C$ 的
最大特征向量（eigenvector）。前 k 个特征向量就是前 k 个主成分。

### 为什么用 SVD 而不是直接特征分解？

数学上等价：对中心化数据 $X_c$，协方差矩阵

$$C = \frac{1}{N-1}X_c^\top X_c$$

做特征分解 $C = V\Lambda V^\top$ 可以得到主成分。但**数值上 SVD 更稳定**：

- 不需要显式构造 $X_c^\top X_c$（条件数会平方，放大误差）
- SVD 一步到位：$X_c = U \cdot \text{diag}(S) \cdot V^\top$

其中 $V^\top$ 的每一行就是一个主成分方向，$S^2/(N-1)$ 就是对应的方差
（特征值）。

### 符号歧义

特征向量的方向不唯一——$v$ 和 $-v$ 张成同一个子空间，LAPACK 不同版本可能
给不同符号。为了判分一致，本题约定：**每个主成分的第一个非零元素为正**，
否则整体翻号。投影列也要同步翻。

## 参考实现

```python
import numpy as np

def _fix_sign(components, projected):
    eps = 1e-12
    for i in range(components.shape[0]):
        row = components[i]
        nonzero = np.abs(row) > eps
        if not nonzero.any():
            continue
        first = np.argmax(nonzero)           # 第一个 |值| > eps 的位置
        if row[first] < 0:
            components[i] = -components[i]
            projected[:, i] = -projected[:, i]

def pca(X, n_components):
    N = X.shape[0]
    X_c = X - X.mean(axis=0, keepdims=True)                # 中心化

    U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    components = Vt[:n_components].copy()                   # (k, D)
    explained_var = (S[:n_components] ** 2) / (N - 1)       # (k,)
    projected = X_c @ components.T                          # (N, k)

    _fix_sign(components, projected)
    return components, explained_var, projected
```

## 关键点

1. **`full_matrices=False` 省时省内存**：不加的话 SVD 会算完整的
   `U: (N, N)` 和 `Vt: (D, D)`，对 N 或 D 很大时巨浪费。加了只算
   `(N, min(N,D))` 和 `(min(N,D), D)`，前 k 个切片就是我们要的。

2. **方差除以 `(N-1)` 不是 `N`**：样本协方差用 `ddof=1`（无偏估计），所以
   特征值 = $S^2 / (N-1)$。写成 $S^2 / N$ 在 N 小时差距明显，判分会挂。

3. **中心化不能忘**：PCA 的理论推导假设数据均值为 0。如果跳过 `X - mean`，
   第一个主成分会指向数据的均值方向，完全错误。

4. **`projected` 和 `components` 必须同步翻号**：`projected = X_c @
   components.T`，components 翻号后 projected 对应列也要翻，否则投影坐标的
   正负方向就和主成分不一致了。

5. **延伸**：SVD 返回的 $S$ 已经降序排列，所以前 k 个自然是最大的 k 个
   方差方向，不需要再排序。PCA 的常见衍生指标有 explained variance ratio
   （每个主成分占总方差的比例）和 cumulative explained variance（选几维能
   保留 95% 信息）——这些可以直接从 `explained_var` 计算得到。
