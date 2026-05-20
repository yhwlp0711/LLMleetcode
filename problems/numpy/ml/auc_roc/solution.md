# 解题思路：ROC 曲线与 AUC

## 核心思路：从「按分数降序逐个加入」看 ROC

ROC 曲线本质上是：**逐渐降低分类阈值**，观察 TPR 和 FPR 如何变化。

- 阈值 = +∞：所有样本预测为负 → TP=FP=0 → (0, 0)
- 阈值 = -∞：所有样本预测为正 → TP=P, FP=N → (1, 1)
- 中间：每降低一次阈值（即把一个新样本「视为正预测」）：
  - 如果它真是正样本 → TP +1 → TPR 上升一格
  - 如果它真是负样本 → FP +1 → FPR 右移一格

所以 ROC 曲线就是「按分数降序遍历样本，画的一条阶梯」。

## 参考实现

```python
def auc_roc(y_true, y_score):
    order = np.argsort(-y_score, kind="stable")     # 降序
    y_sorted = y_true[order].astype(np.float64)

    tp = np.cumsum(y_sorted)                         # 累加正样本数
    fp = np.cumsum(1.0 - y_sorted)                   # 累加负样本数

    P = y_sorted.sum()                               # 正样本总数
    N = len(y_sorted) - P                             # 负样本总数

    tpr = np.concatenate([[0.0], tp / P])
    fpr = np.concatenate([[0.0], fp / N])

    auc = float((0.5 * (tpr[1:] + tpr[:-1]) * np.diff(fpr)).sum())   # 梯形积分
    return fpr, tpr, auc
```

## 关键技巧

### 1. `argsort(-y_score)` 实现降序

NumPy 的 `argsort` 默认升序，加负号是降序最简洁的写法。`kind="stable"`
保证并列时按原顺序，跟参考实现一致。

### 2. `cumsum` 一步算 TP/FP 序列

`y_sorted` 是排好序的 0/1，`cumsum` 直接给出「前 k 个里有几个 1」。
`1 - y_sorted` 翻转后 cumsum 就是 0 的累积数 = FP。

这种 cumsum 技巧把 O(N²) 朴素双循环降到 O(N log N)（瓶颈在 sort）。

### 3. 梯形积分

梯形法：`∫ y dx ≈ Σ 0.5 * (y_i + y_{i+1}) * (x_{i+1} - x_i)`

向量化一行：

```python
auc = (0.5 * (tpr[1:] + tpr[:-1]) * np.diff(fpr)).sum()
```

NumPy 1.x 有 `np.trapz(y, x)` 内置函数；2.x 改名为 `np.trapezoid`。为了
跨版本兼容，本题直接手写公式。

## 易错点

### 1. 头部要加 `(0, 0)`

```python
tpr = np.concatenate([[0.0], tp / P])
fpr = np.concatenate([[0.0], fp / N])
```

不加这个，曲线起点就是 `(fp[0]/N, tp[0]/P)`，跟原点不连续，AUC 算出来偏
小。**经典 bug 之一**。

末尾不需要手动加 `(1, 1)`，因为 cumsum 的最后一个元素自然是 `P` 或 `N`。

### 2. 「随机预测器 AUC=0.5」性质

如果所有 `y_score` 都相同（比如全 1.0），ROC 曲线退化成对角线，AUC=0.5。
这种情况 cumsum 仍能正确处理（排序后 0/1 按原顺序排列，梯形面积刚好
0.5）。**这是一个很好的属性测试**。

### 3. 「完美预测器 AUC=1.0」性质

如果 `y_score = y_true`，所有正样本排在最前面，先涨满 TPR 再涨 FPR，曲
线走 `(0,0) → (0,1) → (1,1)`，AUC=1.0。

## 复杂度

O(N log N) —— 瓶颈在 sort。cumsum、梯形求和都是 O(N)。

## sklearn 的实现差异

`sklearn.metrics.roc_curve` 会**去重阈值**（在分数相同的连续位置只输出
一个点），所以输出的 fpr/tpr 数组长度可能少于 N+1。本题为了简化，**不
去重**，输出长度恰好 N+1。

如果想跟 sklearn 严格对齐，可以在 `np.diff(sorted_score) != 0` 的位置做
indexing —— 但本题不要求。
