# 解题思路：KMeans 聚类

## 算法骨架（Lloyd / 经典 EM 迭代）

每轮重复两步：

1. **E 步（分配）**：每个样本归到最近质心 → 得到 `labels`
2. **M 步（更新）**：每个簇的新质心 = 该簇样本的均值

## 参考实现

```python
def _pairwise_sq_dist(X, C):
    return ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=-1)   # (N, K)

def _assign(X, centroids):
    return _pairwise_sq_dist(X, centroids).argmin(axis=1).astype(np.int64)

def kmeans(X, init_centroids, max_iter, tol=1e-6):
    centroids = init_centroids.copy()
    K = centroids.shape[0]

    for _ in range(max_iter):
        labels = _assign(X, centroids)
        new_centroids = centroids.copy()
        for k in range(K):
            mask = labels == k
            if mask.any():
                new_centroids[k] = X[mask].mean(axis=0)
            # 空簇：保留旧质心

        shift = np.abs(new_centroids - centroids).max()
        centroids = new_centroids
        if shift < tol:
            break

    return centroids, _assign(X, centroids)
```

## 三个关键点

### 1. 平方距离向量化

`X[:, None, :] - C[None, :, :]` 用广播得到 `(N, K, D)` 的差值张量，求平方
和得 `(N, K)`。比写循环快两个数量级。

**省略 sqrt 是有意为之** —— argmin 顺序不变，少一个数学函数调用。

**内存警告**：`(N, K, D)` 张量在 N、K、D 都大时容易爆。这道题规模小所以
没事。真实代码用 `(a-b)² = a² - 2ab + b²` 技巧（参考 tensor_ops 题）：

```python
def _pairwise_sq_dist_fast(X, C):
    x2 = (X ** 2).sum(1, keepdims=True)        # (N, 1)
    c2 = (C ** 2).sum(1)                        # (K,)
    return x2 - 2 * X @ C.T + c2                # (N, K) via broadcast
```

### 2. 空簇怎么办？

每轮 update 前 copy 旧质心，只对**非空簇**赋新质心：

```python
new_centroids = centroids.copy()
for k in range(K):
    mask = labels == k
    if mask.any():
        new_centroids[k] = X[mask].mean(axis=0)
    # else: 跳过，保留旧值
```

**为什么不直接重新初始化空簇？** 因为本题要求确定性可判分 ——「重新随机」
会引入 RNG 状态依赖。生产里常见做法是把空簇移到「最远点」（FurthestPoint
heuristic），但那也是 init 策略的一种，本题简化。

### 3. 收敛判据

```python
shift = np.abs(new_centroids - centroids).max()
if shift < tol:
    break
```

用「质心变化的最大值」而不是「loss」—— 后者要计算成本高，前者一行搞定。
tol 默认 `1e-6`，对一般问题足够。

## 易错点

### 1. `argmin(axis=1)` vs `axis=0`

`pairwise_sq_dist(X, C)` 的 shape 是 `(N, K)`，我们要**对每个样本**找最
近质心 → 在 K 维上 argmin → `axis=1`。写反了会得到 `(K,)`，完全错。

### 2. 第一次 `labels` 在循环外也要算

参考实现里循环结束后又做了一次 `_assign(X, centroids)` —— 因为循环内的
最后一次 `labels` 是用「上一轮的 centroids」算的；最终质心已经更新，要
重新分配一次。

### 3. `int64` dtype

题目要求 labels 是 `int64`。`np.argmin` 在不同 NumPy 版本默认可能是 `int32`
（特别是 Windows）。显式 `.astype(np.int64)` 保险。

## 复杂度

每轮：O(N · K · D)。`max_iter` 轮。整体 O(NKDI)，跟 K 线性相关，是
unsupervised 里最便宜的算法之一。
