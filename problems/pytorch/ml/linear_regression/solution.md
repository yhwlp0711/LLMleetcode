# 解题思路：线性回归（PyTorch Autograd 版）

## 与 NumPy 版的核心差异

- **不需要手算梯度公式**，调用 `loss.backward()`，autograd 引擎自动算。
- **参数更新必须在 `torch.no_grad()` 里做**，否则更新这个操作会被记进计算
  图，下一次 backward 报错。
- **梯度要手动清零**，否则会累加（PyTorch 的设计选择，方便 gradient
  accumulation，但 vanilla SGD 必须清）。

## 参考实现

```python
def fit_predict(X_train, y_train, X_test, *, lr, epochs):
    _, D = X_train.shape
    w = torch.zeros(D, requires_grad=True)
    b = torch.zeros((), requires_grad=True)   # 标量张量

    for _ in range(epochs):
        y_hat = X_train @ w + b
        loss = ((y_hat - y_train) ** 2).mean()
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
            w.grad.zero_()
            b.grad.zero_()

    with torch.no_grad():
        y_pred = X_test @ w + b
    return w.detach(), float(b.item()), y_pred.detach()
```

## 三个易错点

### 1. `b = 0.0` 不行，必须用 `torch.zeros(())` 或 `torch.tensor(0.0, requires_grad=True)`

`requires_grad` 只对张量有意义。如果 `b` 是 Python float，`b -= lr * b.grad`
会报 `AttributeError: 'float' object has no attribute 'grad'`。

### 2. `mean()` vs `sum() / N`

`((y_hat - y) ** 2).mean()` 内部已经除以 `N`，所以梯度公式里就**没有** `2/N`
那个 N 了 —— autograd 直接帮你算对。如果写成 `.sum()`，梯度就缺一个 `1/N`
因子，行为变成「真•梯度下降」而不是「平均梯度下降」，需要把 `lr` 改小。
参考实现用 `.mean()`，跟 NumPy 版的数学等价。

### 3. 返回值必须 `.detach()`

不 detach 的话，外面拿到的张量还携带计算图。要么外面接力 backward（不
是本题语义），要么内存常驻直到该 graph 被 GC。养成「函数边界 detach」的
习惯。

## 为什么不能用 `torch.optim`？

`torch.optim.SGD` 内部就是上面这一套（外加可选的 weight decay / momentum）。
本题就是练这套内部流程，所以禁用。生产代码里当然用 `optim.SGD(params, lr=...)`
+ `optimizer.zero_grad()` + `optimizer.step()`，简洁不易错。
