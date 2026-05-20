# KMeans 聚类

实现经典 Lloyd 算法的 KMeans。**为了判分可复现，初始质心由调用方传入**
（不实现 random / k-means++ init），用户只负责 assign + update 的迭代逻辑。

## 函数签名

```python
def kmeans(
    X: np.ndarray,           # (N, D) 数据
    init_centroids: np.ndarray,  # (K, D) 初始质心（外部给）
    max_iter: int,
    tol: float = 1e-6,       # 质心变化小于此值则提前停止
) -> tuple[np.ndarray, np.ndarray]:
    """
    返回:
        centroids: (K, D)  最终质心
        labels:    (N,)    每个样本所属簇的 id (int64)
    """
```

## 算法（Lloyd）

每轮迭代：

1. **分配步**：每个样本分到**最近**的质心（按欧氏距离 L2）。
2. **更新步**：每个簇的新质心 = 该簇所有样本的均值。
3. **收敛检查**：如果 `max |new_centroids - old_centroids| < tol`，停止。

最多跑 `max_iter` 轮。

## 边界情况：空簇

如果某个簇在某轮**没有被任何样本分配到**（空簇），保留它的旧质心
（不要 nan，也不要重新初始化）。

## 说明

- 输入 `np.float64`。
- 返回的 `labels` 必须是 `np.int64`。
- 距离比较用**欧氏距离的平方**就够（避免 sqrt），结果等价。
- 容差 `atol=1e-8`，质心和 labels 都要对。
