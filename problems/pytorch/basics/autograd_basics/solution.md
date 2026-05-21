# 解题思路：PyTorch Autograd 基础

## 1. `grad_of_scalar`

要点：**不要污染原 `x`**（题目要求）。常见错误是直接 `x.requires_grad_(True)`
原地改入参。正确做法是 `detach + clone + requires_grad_(True)` 做一个隔离
副本：

```python
def grad_of_scalar(x, fn):
    z = x.detach().clone().requires_grad_(True)
    y = fn(z)
    (grad,) = torch.autograd.grad(y, z, create_graph=False)
    return grad
```

`torch.autograd.grad(outputs, inputs)` 返回 inputs 的梯度元组，比
`y.backward()` 更函数式（不依赖 `.grad` 属性）。

> 也可以用 `y.backward()` + `z.grad`，但要先 `z.grad = None` 清零，写起来
> 更脏一些。

## 2. `numerical_jacobian`

中心差分公式：

$$J[i, j] \approx \frac{f\_i(x + \epsilon e\_j) - f\_i(x - \epsilon e\_j)}{2\epsilon}$$

需要循环遍历输入的每一维 `j`（这道题**允许**循环，因为禁用 autograd 后没
有别的办法）。

```python
def numerical_jacobian(fn, x, eps=1e-4):
    x0 = x.detach().clone()
    y0 = fn(x0.clone())
    n, m = x0.numel(), y0.numel()
    J = torch.zeros(m, n, dtype=x0.dtype)
    flat = x0.reshape(-1)
    for j in range(n):
        e = torch.zeros_like(flat)
        e[j] = eps
        y_plus = fn((flat + e).reshape(x0.shape).clone()).reshape(-1)
        y_minus = fn((flat - e).reshape(x0.shape).clone()).reshape(-1)
        J[:, j] = (y_plus - y_minus) / (2 * eps)
    return J
```

**关键点**：

1. **双精度**：题目用 `float64`，否则中心差分误差太大测不过。
2. **`eps` 选择**：太小会被浮点误差吞，太大就不再是「微分近似」。`1e-5` ~
   `1e-4` 一般最稳。
3. **每次调用前 `.clone()`**：避免 `fn` 做 in-place 修改污染下一次。

## 3. `sgd_minimize`

经典 autograd 训练循环模板：

```python
def sgd_minimize(fn, x0, lr, steps):
    x = x0.detach().clone().requires_grad_(True)
    for _ in range(steps):
        y = fn(x)
        y.backward()
        with torch.no_grad():
            x -= lr * x.grad     # 必须在 no_grad 里，否则进入计算图
            x.grad.zero_()        # 不清零会累加
    return x.detach()
```

**易错点**：

1. **`with torch.no_grad():`** 包住参数更新 —— 否则 `x -= lr * x.grad` 会
   被记进计算图，下次 backward 时会出错。
2. **梯度清零**：`x.grad.zero_()`。PyTorch 的 `.backward()` 是**累加**到
   `.grad`，不清零的话第 2 步开始梯度就是错的。等价写法 `x.grad = None`
   也行，但 `zero_()` 复用内存稍微快一点。
3. **返回值要 `detach()`**：避免外部接收者还能反向传播到内部计算图。

## 为什么所有 `x = x.detach().clone()`？

`detach()` 切断计算图，`clone()` 复制数据。组合起来就是「拿到一份纯数据的
副本」。然后再 `requires_grad_(True)` 让它成为新的叶子张量。这是把外部传
入的张量纳入 autograd 而**不影响外部**的标准模式。
