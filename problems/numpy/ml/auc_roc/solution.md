# 解题思路：ROC 曲线与 AUC

## 一句话思路

想象把分类阈值从 +∞ 一路降到 -∞：每降低一次就多把一个样本判为「正」，
观察真阳率 TPR 和假阳率 FPR 怎么变——把这些点连起来就是 ROC 曲线，曲线
下的面积就是 AUC（Area Under ROC Curve，ROC 曲线下面积）。整题用「按分数
降序排序 + 累加（cumsum）+ 梯形积分」三步搞定。

## 从直觉到公式

### ROC 是「逐渐降阈值」画出的阶梯

分类器给每个样本一个分数，我们用一个阈值把它切成正/负。阈值越高越严格：

- 阈值 = +∞：谁都不判为正 → TP=FP=0 → 点 `(0, 0)`。
- 阈值 = -∞：全判为正 → TP=P、FP=N → 点 `(1, 1)`。
- 中间每降低一次阈值，就把「下一个最高分的样本」纳入正预测：
  - 它真是正样本 → TP+1 → TPR 上升一格；
  - 它真是负样本 → FP+1 → FPR 右移一格。

所以只要**按分数降序遍历样本**，就能一步步画出这条阶梯。

### 用 cumsum 一次算出整条曲线

把标签按分数降序排好后得到 0/1 序列 `y_sorted`。此时：

$$\text{TP}[k] = \sum_{i \le k} y\_sorted[i], \quad \text{FP}[k] = \sum_{i \le k} (1 - y\_sorted[i])$$

这正是前缀和——`np.cumsum` 一行搞定。再除以正/负样本总数 $P$、$N$ 得到
TPR、FPR。头部补上 `(0, 0)`（阈值 +∞ 时的起点），末尾 cumsum 自然到达
`(1, 1)`。

### 梯形积分算面积

AUC 用梯形法（trapezoidal rule）累加相邻两点围成的梯形面积：

$$\text{AUC} = \sum_i \tfrac{1}{2}\,(\text{tpr}_{i+1} + \text{tpr}_i)\,(\text{fpr}_{i+1} - \text{fpr}_i)$$

## 参考实现

```python
import numpy as np

def auc_roc(y_true, y_score):
    order = np.argsort(-y_score, kind="stable")   # 降序，稳定排序保并列顺序
    y_sorted = y_true[order].astype(np.float64)

    tp = np.cumsum(y_sorted)                       # 前 k 个里的正样本数
    fp = np.cumsum(1.0 - y_sorted)                 # 前 k 个里的负样本数

    P = y_sorted.sum()
    N = len(y_sorted) - P

    tpr = np.concatenate([[0.0], tp / P])          # 头部补 (0,0)
    fpr = np.concatenate([[0.0], fp / N])

    auc = float((0.5 * (tpr[1:] + tpr[:-1]) * np.diff(fpr)).sum())  # 梯形积分
    return fpr, tpr, auc
```

## 关键点

1. **`argsort(-y_score)` 实现降序**：NumPy 的 `argsort` 默认升序，取负号是
   最简洁的降序写法。`kind="stable"` 保证分数并列时按原顺序处理，和判分
   口径一致。

2. **cumsum 把 O(N²) 降到 O(N)**：`y_sorted` 是排好的 0/1，`cumsum` 直接
   给出「前 k 个里有几个正样本」= TP 序列；`1 - y_sorted` 的 cumsum 就是
   FP 序列。省去了对每个阈值重新数一遍的双重循环，整体瓶颈只在排序
   O(N log N)。

3. **头部必须补 `(0, 0)`**：不补的话曲线起点是 `(fp[0]/N, tp[0]/P)`，与
   原点不连续，AUC 会偏小——这是最常见的一处疏漏。末尾无需手动补 `(1, 1)`，
   因为 cumsum 的最后一项自然等于 $P$（或 $N$）。

4. **两个直觉基准**：分数全相同 → ROC 退化成对角线，AUC=0.5（相当于随机
   猜）；分数完美区分正负 → 曲线走 `(0,0)→(0,1)→(1,1)`，AUC=1.0。判分时
   这两个属性很好用来自检。

5. **延伸**：`sklearn.metrics.roc_curve` 会对相同分数的阈值去重，输出点数
   可能少于 N+1；本题为简化不去重，输出恰好 N+1 个点。AUC 还等价于「随机
   取一对正负样本，正样本分数更高的概率」，这也是它常被用作排序质量指标的
   原因。
