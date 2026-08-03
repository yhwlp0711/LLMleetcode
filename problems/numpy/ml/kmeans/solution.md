# 解题思路：KMeans 聚类

## 一句话思路

KMeans 用一个来回迭代的过程把数据分成 K 簇：**先把每个点分给最近的质心
（分配步），再把每个质心挪到自己那簇的中心（更新步）**，反复直到质心几乎
不动。这就是经典的 Lloyd 算法。难点在于向量化算「点到质心」的距离，以及
处理没分到任何点的「空簇」。

## 拆解思路

### 一轮迭代 = 分配 + 更新

**分配步**：对每个样本，算它到 K 个质心的欧氏距离（L2 距离），选最近的那个
作为它的簇标签。比较距离时用**平方距离**就够——开不开根号不改变谁最近，
省一次 sqrt。

**更新步**：每个簇的新质心 = 该簇所有样本的均值（这是「让簇内平方误差最小」
的最优选择）。

**收敛检查**：若所有质心这一轮的移动量都小于 `tol`，就提前停止。

### 向量化算距离矩阵

用广播（broadcasting）一次算出所有点到所有质心的平方距离：
`X[:, None, :] - C[None, :, :]` 得到 `(N, K, D)` 的差值张量，平方后沿最后
一维求和得 `(N, K)`。然后 `argmin(axis=1)` 就是每个样本最近的质心索引。

### 空簇怎么办

若某个簇这一轮没被任何样本选中（空簇），它的均值会是 NaN。题目要求这时
**保留旧质心**。做法：更新前先 copy 一份旧质心，只对非空簇写入新均值，空簇
自然保持原值。

## 参考实现

```python
import numpy as np

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
            # 空簇：保留旧质心（跳过赋值）

        shift = np.abs(new_centroids - centroids).max()
        centroids = new_centroids
        if shift < tol:
            break

    return centroids, _assign(X, centroids)
```

## 关键点

1. **平方距离向量化 + 省略 sqrt**：`X[:, None, :] - C[None, :, :]` 用广播
   得到 `(N, K, D)` 差值，平方求和即距离平方。因为 argmin 只看相对大小，
   开根号是多余的。注意这个 `(N, K, D)` 张量在 N、K、D 都大时会占很多内存，
   更省内存的写法是用 $(a-b)^2 = a^2 - 2ab + b^2$ 展开（见 `numpy.ml.knn`）。

2. **`argmin(axis=1)` 不能写错轴**：距离矩阵 shape 是 `(N, K)`，我们要为
   **每个样本**（第 0 维）在 K 个质心里挑最近的，所以沿 `axis=1` 求最小。
   写成 `axis=0` 会得到 `(K,)`，含义完全错。

3. **空簇保留旧质心，保证确定性**：更新前 `new_centroids = centroids.copy()`，
   只对有样本的簇写入均值。不重新随机初始化空簇，是因为本题要求可复现判分
   ——引入随机数会破坏确定性。

4. **收敛判据用「质心最大移动量」**：`np.abs(new - old).max() < tol` 比计算
   簇内损失更省事，一行搞定。循环结束后再 `_assign` 一次，是因为最后一轮
   更新过质心，标签要用新质心重新算。

5. **延伸**：本题的初始质心由外部传入以保证可复现；实际中常用 k-means++
   来挑初始点。分配步「找最近质心」本质就是沿某轴做 argmin，和
   `numpy.basics.argmax_along_axis` 是同一类操作。
