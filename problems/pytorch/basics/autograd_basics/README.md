# PyTorch Autograd 基础

练习 PyTorch autograd 的核心生命周期：`requires_grad`、`.backward()`、梯
度累加、`torch.no_grad()`。

## 待实现函数

### 1. `grad_of_scalar(x, fn)`
给定一个叶子张量 `x` 和一个把张量映射到标量张量的函数 `fn`，返回
`df/dx`，shape 与 `x` 相同。

```python
def grad_of_scalar(x: torch.Tensor, fn: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    ...
```

不要修改 `x`。不要假设 `x` 已经 `requires_grad=True`。

### 2. `numerical_jacobian(fn, x, eps=1e-4)`
用**中心差分**估计 `fn: R^n -> R^m` 在 `x` 处的 Jacobian：
`J[i, j] ≈ (fn(x + eps * e_j)[i] - fn(x - eps * e_j)[i]) / (2 * eps)`。

```python
def numerical_jacobian(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """返回 shape 为 (m, n) 的中心差分 Jacobian。"""
```

**禁用 autograd** —— 这是数值参考实现。请用 Python 循环遍历输入维度。

### 3. `sgd_minimize(fn, x0, lr, steps)`
从 `x0` 开始，对标量函数 `fn` 做 `steps` 步 vanilla SGD，学习率 `lr`，返回
最终的 `x`（detach 过的）。

```python
def sgd_minimize(fn, x0: torch.Tensor, lr: float, steps: int) -> torch.Tensor:
    ...
```

在 `torch.no_grad()` 下做参数更新，每步之后清零梯度。**禁用** `torch.optim`。

## 说明

- 所有输入是 `torch.float64`（数值梯度对比需要双精度）。
- 判分：解析梯度与你的 `grad_of_scalar` 对比；autograd 真梯度与你的
  `numerical_jacobian` 对比（容差较宽 `atol=1e-3` ~ `1e-4`）。
