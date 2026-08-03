# 解题思路：PyTorch Autograd 基础

## 一句话思路

这题练的是 PyTorch 自动微分（autograd）的完整生命周期：怎么把张量标记成「需要
算梯度」、怎么反向传播（backprop）拿到梯度、以及在更新参数时怎么**关掉**
autograd。三个小函数分别对应三件事：取解析梯度、用差分手算数值梯度、写一个最朴
素的梯度下降循环。

## 拆解思路

### 1. `grad_of_scalar`：取标量对输入的梯度

autograd 只对「叶子张量（leaf tensor）里 `requires_grad=True` 的那些」记录梯度。
题目要求不能假设 `x` 已经开了梯度，也不许改动 `x`，所以我们先复制一份、再打开开
关：

- `x.detach().clone()`：拷一份和原计算图无关的新张量。
- `.requires_grad_(True)`：让它开始被 autograd 跟踪。

接着 `torch.autograd.grad(y, z)` 直接返回 `dy/dz`，比 `y.backward()` 更干净——它
不会把梯度堆到 `.grad` 属性上，也不用手动清零。它的返回值是个 tuple，所以用
`(grad,) = ...` 解包。

### 2. `numerical_jacobian`：用中心差分手算雅可比

这是一个**不依赖 autograd** 的数值参考，用来验证 autograd 算得对不对。它来自导数
的定义——把输入的第 `j` 个分量往正负两个方向各挪一点点，看输出怎么变：

$$J[i, j] \approx \frac{f(x + \epsilon e_j)_i - f(x - \epsilon e_j)_i}{2\epsilon}$$

这里 $e_j$ 是「只有第 `j` 位是 1」的单位向量。用**中心差分**（两边都挪）而不是单
边差分，是因为它的误差是 $O(\epsilon^2)$，精度高一个量级。因为要逐个分量扰动，
这里**必须**用 Python 循环遍历 `n` 个输入维度。

### 3. `sgd_minimize`：手写梯度下降

最朴素的随机梯度下降（SGD）：每步「前向算 loss → 反向 `.backward()` 拿梯度 → 沿
梯度反方向走一小步」。两个关键约定：

- 参数更新放在 `torch.no_grad()` 里。更新本身是「拿旧参数算新参数」的张量运算，我
  们不希望它也被 autograd 记进计算图，否则图会越滚越大、还会算错。
- 每步之后 `x.grad.zero_()` 清零。PyTorch 的梯度是**累加**的，不清零的话下一步会
  叠加到这一步上。

## 参考实现

```python
def grad_of_scalar(x, fn):
    z = x.detach().clone().requires_grad_(True)   # 拷一份并开启梯度跟踪
    y = fn(z)
    (grad,) = torch.autograd.grad(y, z)           # 直接返回 dy/dz
    return grad


def numerical_jacobian(fn, x, eps=1e-4):
    x0 = x.detach().clone()
    y0 = fn(x0.clone())
    n, m = x0.numel(), y0.numel()
    J = torch.zeros(m, n, dtype=x0.dtype)
    flat = x0.reshape(-1)
    for j in range(n):                            # 逐个输入维度扰动
        e = torch.zeros_like(flat)
        e[j] = eps
        y_plus = fn((flat + e).reshape(x0.shape).clone()).reshape(-1)
        y_minus = fn((flat - e).reshape(x0.shape).clone()).reshape(-1)
        J[:, j] = (y_plus - y_minus) / (2 * eps)  # 中心差分
    return J


def sgd_minimize(fn, x0, lr, steps):
    x = x0.detach().clone().requires_grad_(True)
    for _ in range(steps):
        y = fn(x)
        y.backward()
        with torch.no_grad():                     # 更新不进计算图
            x -= lr * x.grad
            x.grad.zero_()                         # 梯度累加，必须清零
    return x.detach()
```

## 关键点

1. **`detach().clone()` 的意义**：`detach` 切断与原计算图的联系，`clone` 复制数据。
   两者合起来得到一个「干净、可独立求导」的新张量，既满足「不修改 `x`」，又能安全
   地开梯度。

2. **为什么用中心差分而不是单边差分？** 单边差分
   $(f(x+\epsilon)-f(x))/\epsilon$ 的误差是 $O(\epsilon)$；中心差分让一阶误差项相
   互抵消，误差降到 $O(\epsilon^2)$，所以判分时数值梯度能和 autograd 真梯度对得很
   齐。题目用双精度（`float64`）也是为了减小这种数值误差。

3. **梯度会累加**：PyTorch 不会自动清空 `.grad`，这是为了支持「梯度累积」等技巧。
   普通训练里每步更新完必须手动清零，否则梯度越滚越大。

4. **更新为什么要 `no_grad`**：`x -= lr * x.grad` 是一次张量运算，不关掉 autograd
   它会被记进计算图，导致下一次 `backward` 报错或算错。这也是所有手写训练循环的固
   定套路。

5. **延伸**：把这里的手写 SGD 换成 `torch.optim.SGD`、把标量函数换成真实的 MSE 损
   失，就是标准的模型训练流程——见 `pytorch.ml.linear_regression`，那题正是用同一
   套「前向 → backward → no_grad 更新 → 清零」的循环训练线性回归。
