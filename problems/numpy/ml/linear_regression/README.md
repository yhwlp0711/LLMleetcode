# 手撕线性回归（NumPy）

实现一个简单的线性回归模型，用**批量梯度下降**训练。

## 函数签名

```python
def fit_predict(
    X_train: np.ndarray,   # shape (N, D)
    y_train: np.ndarray,   # shape (N,)
    X_test:  np.ndarray,   # shape (M, D)
    *,
    lr: float,
    epochs: int,
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    返回:
        w:        shape (D,)  训练后的权重
        b:        scalar     训练后的偏置
        y_pred:   shape (M,)  用 (w, b) 在 X_test 上的预测
    """
```

## 要求

1. `w` 初始化为全零向量（shape `(D,)`），`b = 0.0`。
2. 使用 **均方误差（MSE）** 损失：
   $$L = \frac{1}{N}\sum_i (\hat{y}_i - y_i)^2$$
3. 使用 **批量梯度下降**（一次更新用全部样本，不分 mini-batch），共 `epochs` 轮：
   - 前向：$\hat{y} = Xw + b$
   - 梯度：$\nabla_w = \frac{2}{N} X^T(\hat{y} - y)$，$\nabla_b = \frac{2}{N}\sum(\hat{y} - y)$
   - 更新：$w \leftarrow w - \text{lr} \cdot \nabla_w$，$b \leftarrow b - \text{lr} \cdot \nabla_b$
4. 训练完后返回 `w`、`b`、以及在 `X_test` 上的预测。

## 说明

- **只用 NumPy**（禁用 `sklearn` / `torch`）。
- 在固定的输入和超参下，训练过程完全确定，所以判分器对 `w`、`b`、`y_pred`
  做严格数值对比（`atol=1e-6`）。

## 提示

参考实现大约 10 行 NumPy。
