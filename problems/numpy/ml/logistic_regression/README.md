# 手撕逻辑回归（NumPy）

实现二分类逻辑回归，用**批量梯度下降**最小化二分类交叉熵损失。

## 函数签名

```python
def fit_predict_proba(
    X_train: np.ndarray,   # (N, D)
    y_train: np.ndarray,   # (N,) 取值 {0, 1}
    X_test:  np.ndarray,   # (M, D)
    *,
    lr: float,
    epochs: int,
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    返回:
        w: (D,)   训练后的权重
        b: scalar 训练后的偏置
        proba_test: (M,) sigmoid(X_test @ w + b)
    """
```

## 要求

1. `w` 初始化为全零向量，`b = 0.0`。
2. 使用 **sigmoid** 激活：$\sigma(z) = 1 / (1 + e^{-z})$。
   **要数值稳定** —— 当 `z` 非常负时 `exp(-z)` 会溢出，需要分支处理或用
   `np.where` + `1 / (1 + exp(-|z|))` 的技巧。
3. 使用 **二分类交叉熵** 损失：
   $L = -\frac{1}{N}\sum_i \bigl[y_i \log p_i + (1-y_i) \log(1 - p_i)\bigr]$
4. 梯度（已经化简过）：
   $\nabla_w = \frac{1}{N} X^T (p - y)$，$\nabla_b = \frac{1}{N} \sum (p - y)$
5. 共 `epochs` 轮 SGD 更新。

## 说明

- **只用 NumPy**（禁用 `sklearn` / `torch`）。
- 判分器对最终的 `w`、`b`、`proba_test` 做严格数值对比（`atol=1e-6`）。
