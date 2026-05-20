# 手撕线性回归（PyTorch Autograd 版）

和 `numpy.ml.linear_regression` 是同一道题，但要求用 **PyTorch 张量 +
autograd** 训练，不再手算梯度。考察 `requires_grad` / `.backward()` / 在
`torch.no_grad()` 下做参数更新的标准流程。

## 函数签名

```python
def fit_predict(
    X_train: torch.Tensor,   # (N, D), float32
    y_train: torch.Tensor,   # (N,),   float32
    X_test:  torch.Tensor,   # (M, D), float32
    *,
    lr: float,
    epochs: int,
) -> tuple[torch.Tensor, float, torch.Tensor]:
    """
    返回:
        w:      (D,) float32，训练后的权重（detach、无 grad）
        b:      Python float
        y_pred: (M,) float32，在 X_test 上的预测
    """
```

## 要求

1. `w` 初始化为全零（shape `(D,)`），`b = 0.0`。两者都要
   `requires_grad=True`。
2. 损失用 **MSE**：`((X @ w + b - y) ** 2).mean()`。
3. **批量梯度下降**共 `epochs` 轮：
   - 前向 → loss → `.backward()` → 在 `torch.no_grad()` 下原地更新 `w`、`b`
     → 把梯度清零。
4. 返回 `(w.detach(), float(b.item()), (X_test @ w + b).detach())`。

## 说明

- 全程在 CPU；不要移到其他设备。
- **禁用** `torch.optim` —— 手写更新流程才是题目要练的东西。
- 判分对比参考实现，`atol=1e-5`。
