# 解题思路：KNN 分类

## 三步走

1. **距离矩阵**：每个测试点 vs 每个训练点 → `(M, N)` 平方距离
2. **k 近邻**：每行 argpartition 取最小的 k 个索引
3. **多数投票**：用 `np.bincount` 数标签出现次数，argmax 取众数

## 参考实现

```python
def knn_predict(X_train, y_train, X_test, k, num_classes):
    # 1. 距离 (a-b)² = a² - 2ab + b²
    x2 = (X_test ** 2).sum(axis=1, keepdims=True)        # (M, 1)
    t2 = (X_train ** 2).sum(axis=1)                       # (N,)
    xt = X_test @ X_train.T                               # (M, N)
    dist = x2 - 2.0 * xt + t2                             # (M, N) 广播

    # 2. argpartition：O(N) 找前 k 个（不要求顺序），比 argsort O(N log N) 快
    idx_k = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]

    # 3. 投票
    M = dist.shape[0]
    preds = np.empty(M, dtype=np.int64)
    for i in range(M):
        votes = np.bincount(y_train[idx_k[i]], minlength=num_classes)
        preds[i] = votes.argmax()
    return preds
```

## 关键技巧

### 1. 距离矩阵的「乘积展开」

`(a-b)² = a² - 2ab + b²` 拆开后只需要一次 `M × N` 的矩阵乘 + 两次平方和，
内存 O(MN)。直接广播 `(M, 1, D) - (1, N, D)` 得到 `(M, N, D)` 也对，但
内存 O(MND) —— D 大时爆。

### 2. `np.bincount(arr, minlength=K)`

数组中每个值出现的次数，返回长度 `K` 的计数数组。比 `Counter` 快太多。
**注意必须设 `minlength=num_classes`**，否则如果某次取到的 k 个邻居没有
覆盖到最高的类别 id，输出长度不够，索引错位。

### 3. `argmax` 自然处理"并列取最小 id"

NumPy 的 `argmax` 在并列时返回**最小索引**。题目要求正好如此，免去额外
处理。如果要求"取最大"或者别的 tie-breaking 规则，要自己写。

### 4. 为什么不用 `argsort`？

`np.argsort(dist, axis=1)[:, :k]` 也对，但 `argsort` 是 O(N log N)；
`argpartition` 只保证前 k 个是「最小的 k 个」（顺序不一定），是 O(N)。
当 N 远大于 k 时（典型 KNN 场景），`argpartition` 快很多。

排序后是否还重要？对**多数投票**不重要 —— 投票只看是哪些标签，不看顺序。

## 易错点

### 1. y_train 是 `int64`

`np.bincount` 要求**非负整数**输入。如果你不小心把 `y_train` 当 float 处
理（比如除以一个常数），bincount 会报错。

### 2. M 维循环可以保留

题目允许在 M 维上循环（因为 bincount 每行独立）。如果想全向量化也可以
（构造 one-hot + sum），但 M 一般是 batch size 不会太大，循环 + bincount
反而代码更清晰。

### 3. `k=1` 退化为「最近邻」

`k=1` 时投票只有 1 票，等价于直接返回最近训练点的标签。代码里不需要特殊
处理。

## 复杂度

- 距离矩阵：O(MND)
- argpartition：O(MN)
- 投票：O(Mk)

总计 O(MND)。当 N 上百万时不可行，工业里用 ANN 索引（FAISS / HNSW）加
速。本题规模小，朴素算法够用。
