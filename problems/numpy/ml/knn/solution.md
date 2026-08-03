# 解题思路：KNN 分类

## 一句话思路

KNN（K-Nearest Neighbors，K 近邻）没有训练阶段：预测时对每个测试点，找出
训练集里离它最近的 K 个邻居，让它们**投票**决定类别。三步走——算距离矩阵、
取最近 K 个、多数投票。难点在于把距离矩阵一次性向量化算出来。

## 拆解思路

### 用「乘积展开」向量化算距离矩阵

要算每个测试点到每个训练点的欧氏距离平方，共 `(M, N)` 个。直接广播
`(M, 1, D) - (1, N, D)` 得 `(M, N, D)` 也行，但 D 大时内存爆炸。更省的
办法是把平方距离展开：

$$\|x - t\|^2 = \|x\|^2 - 2\,x^\top t + \|t\|^2$$

这样只需一次 `(M, D) @ (D, N)` 的矩阵乘算出所有内积，再加上两边的平方和
（靠广播），内存只要 O(MN)。比较距离不用开根号，平方距离排序结果一样。

### 取最近 K 个：argpartition

`np.argpartition(dist, kth=k-1, axis=1)` 只保证「前 k 个是最小的 k 个」
（内部不排序），复杂度 O(N)，比全排序 `argsort` 的 O(N log N) 快。因为
投票只关心是哪 k 个邻居、不关心它们之间的顺序，argpartition 足够。

### 多数投票：bincount + argmax

对每个测试点，用 `np.bincount` 数出这 k 个邻居里每个类别出现几次，
`argmax` 取票数最多的类。NumPy 的 `argmax` 在并列时返回**最小索引**，
正好满足题目「并列取 id 较小的类别」。

## 参考实现

```python
import numpy as np

def knn_predict(X_train, y_train, X_test, k, num_classes):
    x2 = (X_test ** 2).sum(axis=1, keepdims=True)   # (M, 1)
    t2 = (X_train ** 2).sum(axis=1)                  # (N,)
    xt = X_test @ X_train.T                          # (M, N)
    dist = x2 - 2.0 * xt + t2                         # (M, N) 平方距离

    idx_k = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]  # 最近 k 个索引

    M = dist.shape[0]
    preds = np.empty(M, dtype=np.int64)
    for i in range(M):
        votes = np.bincount(y_train[idx_k[i]], minlength=num_classes)
        preds[i] = votes.argmax()                    # 并列取最小 id
    return preds
```

## 关键点

1. **乘积展开省内存**：`(a-b)² = a² - 2ab + b²` 把距离矩阵拆成「一次矩阵乘
   + 两个平方和向量的广播相加」，内存 O(MN)。直接三维广播 `(M, 1, D) -
   (1, N, D)` 结果对，但中间张量是 O(MND)，D 大时会爆。

2. **`bincount` 必须设 `minlength=num_classes`**：否则若这 k 个邻居没覆盖到
   编号最大的类别，返回的计数数组长度不够，后续 argmax 的索引就错位了。
   显式指定长度保证输出对齐所有类别。

3. **`argmax` 天然处理并列取最小 id**：NumPy 的 argmax 在票数相同时返回
   最小索引，与题目要求一致，不用额外写 tie-breaking。若要求别的规则就得
   自己处理。

4. **投票循环 M 次可以接受**：距离矩阵已向量化，投票部分每行独立、逻辑
   清晰，M 通常是 batch 大小不会太大，循环 + bincount 反而比强行全向量化的
   one-hot 求和更好读。

5. **延伸**：`k=1` 退化成最近邻。整体复杂度 O(MND)，N 上百万时不可行，
   工业中改用近似最近邻索引（FAISS / HNSW）。这里的成对距离矩阵思路和
   `numpy.ml.kmeans` 的分配步是同一套。
