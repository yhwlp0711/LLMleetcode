# 解题思路：线性回归（PyTorch Autograd 版）

## 一句话思路

和 `numpy.ml.linear_regression` 是同一道题，只是这次**不用手推梯度公式**——把参数
标成「需要梯度」，调 `loss.backward()`，PyTorch 的自动微分（autograd）帮你算好。
难点全在训练循环的三个约定：参数更新要放进 `torch.no_grad()`、梯度每步要清零、返
回前要 `detach()`。

## 拆解思路

### 模型和损失

线性回归的预测是 $\hat y = Xw + b$，损失用均方误差（MSE）：

$$\mathcal{L} = \frac{1}{N}\sum_i (\hat y_i - y_i)^2$$

按题目要求，`w` 初始化为全零、`b` 初始化为标量 0，两者都开
`requires_grad=True`，让 autograd 跟踪它们。

### 训练循环三步走

每一轮（epoch）做批量梯度下降：

1. **前向 + 算 loss**：`y_hat = X @ w + b`，再算 MSE。
2. **反向**：`loss.backward()` 把梯度填进 `w.grad`、`b.grad`。
3. **更新参数**：沿梯度反方向走一步 `w -= lr * w.grad`，再把梯度清零。

第 3 步必须包在 `torch.no_grad()` 里，因为参数更新本身是张量运算，不关掉 autograd
它会被记进计算图，下一轮 `backward` 就会报错或算错。梯度清零也不能忘——PyTorch 的
梯度是**累加**的，不清零下一轮会叠加到这一轮上。

## 参考实现

```python
def fit_predict(X_train, y_train, X_test, *, lr, epochs):
    _, D = X_train.shape
    w = torch.zeros(D, requires_grad=True)
    b = torch.zeros((), requires_grad=True)     # 标量张量，不是 Python float

    for _ in range(epochs):
        y_hat = X_train @ w + b
        loss = ((y_hat - y_train) ** 2).mean()  # MSE
        loss.backward()
        with torch.no_grad():                   # 更新不进计算图
            w -= lr * w.grad
            b -= lr * b.grad
            w.grad.zero_()                       # 梯度累加，必须清零
            b.grad.zero_()

    with torch.no_grad():
        y_pred = X_test @ w + b
    return w.detach(), float(b.item()), y_pred.detach()
```

## 关键点

1. **`b` 必须是张量，不能是 `0.0`**：`requires_grad` 只对张量有意义。如果 `b` 是
   Python float，`b -= lr * b.grad` 会报 `'float' object has no attribute 'grad'`。
   用 `torch.zeros(())` 建一个标量张量（0 维张量）。

2. **`.mean()` vs `.sum()`**：`.mean()` 内部已经除以 `N`，所以 autograd 算出的梯度
   自带 `1/N` 因子，跟 NumPy 版手推的公式数学等价。若写成 `.sum()`，梯度会大 `N`
   倍，得把 `lr` 相应改小才等价。

3. **参数更新为什么要 `no_grad`**：`w -= lr * w.grad` 是张量运算，不关 autograd 会
   被记进计算图，导致梯度累积错误、内存膨胀，下一轮 `backward` 直接报错。这是所有
   手写训练循环的固定套路。

4. **返回值要 `detach()`**：不 detach 的话外面拿到的张量还挂着计算图，既可能被误
   用去反向传播，也会让整张图常驻内存直到被回收。在函数边界 detach 是好习惯。

5. **延伸**：为什么禁用 `torch.optim`？因为 `optim.SGD` 内部就是上面这套流程（外加
   momentum、weight decay 等）。本题就是练这套内部机制。同样的「前向 → backward →
   no_grad 更新 → 清零」循环也出现在 `pytorch.basics.autograd_basics` 的
   `sgd_minimize` 里。
