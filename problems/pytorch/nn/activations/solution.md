# 解题思路：激活函数

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

## 4. `swiglu`

```python
def swiglu(x, gate):
    return silu(gate) * x
```

**关键点是顺序**：`silu` 套在 `gate` 上，不是 `x` 上。LLaMA FFN 实际写
法是：

```python
y = silu(self.gate_proj(x)) * self.up_proj(x)
out = self.down_proj(y)
```

参数 `gate` 对应 `gate_proj(x)`，`x` 对应 `up_proj(x)`。本题只考门控运算，
不涉及投影矩阵（那是 `swiglu_ffn` 的事）。

## 5. `geglu`

```python
def geglu(x, gate):
    return gelu_exact(gate) * x
```

跟 SwiGLU 同模板，只是把 SiLU 换成 GELU。

## 「门控」是什么意思？

普通激活：$\sigma(z)$ —— 输入只有一个张量。

门控激活：$\sigma(\text{gate}) \cdot x$ —— 输入是两个张量，一个充当「值」，
一个被激活后充当「门」（控制让多少值通过）。理论上模型容量更强：每个位置
有两条独立的信息通路，可以学到「条件性」表征。

LLaMA、PaLM 等现代 LLM 普遍把 FFN 从经典的 `Linear → activation → Linear`
（2 个矩阵）换成 `Linear → activation × Linear → Linear`（3 个矩阵）。因为
多了一倍的输入投影，hidden dim 通常会缩到 2/3 来保持参数量大致相当。

## 6. `sigmoid`

```python
def sigmoid(x):
    pos = x >= 0
    neg = ~pos
    out = torch.empty_like(x)
    out[pos] = 1.0 / (1.0 + torch.exp(-x[pos]))
    exp_x = torch.exp(x[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    return out
```

**为什么要分两支？** 朴素 `1/(1+exp(-x))`：
- `x = -1000` → `exp(1000)` = inf → 结果 `1/inf = 0`（碰巧对了，但中间溢出了）
- `x = 1000` → `exp(-1000)` = 0 → 结果 `1/1 = 1`（没问题）

分支法让 `exp` 的参数**永远 ≤ 0**：
- `x ≥ 0`：`exp(-x)` ∈ (0, 1]，安全
- `x < 0`：`exp(x)` ∈ (0, 1]，安全

两种写法数学等价：`e^x / (1+e^x) = 1 / (1+e^{-x})`。

## 7. `softmax`

```python
def softmax(x, dim=-1):
    m = x.max(dim=dim, keepdim=True).values
    e = (x - m).exp()
    return e / e.sum(dim=dim, keepdim=True)
```

**数值稳定性**：先减去 max 后所有指数 ≤ 0，`exp` ∈ (0, 1]，不溢出。
数学上 softmax 平移不变：`softmax(x) == softmax(x - c)` 对任意常数 `c`。

**`keepdim=True` 是关键**：让 max / sum 结果保持维度，方便广播回原 shape。

## 数值稳定性

sigmoid 和 softmax 的稳定性是面试高频问题。核心原则：**让 `exp` 的参数
永远不超过 0**（或者让分子分母的 exp 参数范围一致），避免 overflow。

对于 fp16，`exp(x)` 在 `x > 11` 就 overflow。工业代码通常用 `torch.sigmoid`
/ `F.softmax`（它们内部已处理好），但面试时要能手写稳定版本。
