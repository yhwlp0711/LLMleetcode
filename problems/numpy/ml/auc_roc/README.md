# ROC 曲线与 AUC

实现二分类的 ROC 曲线和 AUC（Area Under ROC Curve）计算。

## 函数签名

```python
def auc_roc(
    y_true: np.ndarray,   # (N,) 取值 {0, 1}, int64
    y_score: np.ndarray,  # (N,) 浮点分数（概率或 logit 都可）
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    返回:
        fpr: (M,) False Positive Rate，按阈值递减排好
        tpr: (M,) True Positive Rate，对应阈值
        auc: scalar，ROC 曲线下的面积
    """
```

## 算法

1. 按 `y_score` **降序**排序。
2. 沿着排序遍历，逐渐降低阈值：
   - 累计 TP（真阳性）和 FP（假阳性）
   - `TPR = TP / P`，`FPR = FP / N`（P 是正样本总数，N 是负样本总数）
3. 在 `(FPR, TPR)` 坐标里画曲线 —— 即一系列点。
4. 用**梯形面积法**累加 AUC。

## 返回的 ROC 点要求

返回的 `fpr` / `tpr` 必须**以 `(0, 0)` 起点、`(1, 1)` 终点**，中间点按阈值
递减（即 FPR 单调不减、TPR 单调不减）。

具体生成方式：

```
点 0：阈值 = +inf，TP=FP=0 → (FPR=0, TPR=0)
点 i：阈值 = sorted_score[i-1]，累加该样本到 TP/FP
点 M-1：阈值 = -inf，TP=P, FP=N → (FPR=1, TPR=1)
```

## 说明

- 输入分数可重复；并列时按**原顺序**处理（不要求 tie-aware AUC，简化）。
- **禁用 sklearn**。
- 假设至少有 1 个正样本和 1 个负样本。
- 容差 `atol=1e-10`。
