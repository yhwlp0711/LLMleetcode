# 解题思路：激活函数（SiLU / GELU）

## 1. `silu`

定义直接照抄即可。`torch.sigmoid(x) = 1 / (1 + exp(-x))`：

```python
def silu(x):
    return x * torch.sigmoid(x)
```

**为什么 SiLU 比 ReLU 好？** ReLU 在 `x < 0` 处梯度恒为 0（dead neuron 问
题），SiLU 是平滑函数，处处可微，训练更稳定。

## 2. `gelu_exact`

公式：$\text{GELU}(x) = 0.5 x (1 + \text{erf}(x / \sqrt{2}))$

```python
from math import sqrt
def gelu_exact(x):
    return 0.5 * x * (1.0 + torch.erf(x / sqrt(2.0)))
```

`torch.erf` 是 [error function](https://en.wikipedia.org/wiki/Error_function)，
PyTorch 内置，无需自己实现。

## 3. `gelu_tanh`

公式直接照抄：

```python
from math import pi, sqrt
def gelu_tanh(x):
    c = sqrt(2.0 / pi)
    return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * x.pow(3))))
```

`x.pow(3)` 等价于 `x ** 3` 但更明确。

**为什么有 tanh 近似？** `erf` 在早期硬件 / 低精度算子里实现昂贵；用
`tanh` + 三次多项式近似在 fp16 下足够精确且更快。今天硬件足够强，但很多
模型（GPT-2、GPT-3）训出来时用的就是 tanh 近似版，加载预训练权重时必须
保持一致才能数值对齐。
