# 解题思路：门控激活函数

## 1. `swiglu`

SiLU 内联写在里面即可（`torch.sigmoid` 允许用；被禁的是 `F.silu`）：

```python
def swiglu(x, gate):
    silu_gate = gate * torch.sigmoid(gate)
    return silu_gate * x
```

**关键点是顺序**：SiLU 套在 `gate` 上，不是 `x` 上。LLaMA FFN 实际写法是：

```python
y = silu(self.gate_proj(x)) * self.up_proj(x)
out = self.down_proj(y)
```

参数 `gate` 对应 `gate_proj(x)`，`x` 对应 `up_proj(x)`。本题只考门控运算，不涉及投影矩阵（那是 `swiglu_ffn` 的事）。

## 2. `geglu`

```python
from math import sqrt
def geglu(x, gate):
    gelu_gate = 0.5 * gate * (1.0 + torch.erf(gate / sqrt(2.0)))
    return gelu_gate * x
```

跟 SwiGLU 同模板，只是把 SiLU 换成精确版 GELU（`torch.erf`）。

## 「门控」是什么意思？

普通激活：$\sigma(z)$ —— 输入只有一个张量。

门控激活：$\sigma(\text{gate}) \cdot x$ —— 输入是两个张量，一个充当「值」，一个被激活后充当「门」（控制让多少值通过）。理论上模型容量更强：每个位置有两条独立的信息通路，可以学到「条件性」表征。

LLaMA、PaLM 等现代 LLM 普遍把 FFN 从经典的 `Linear → activation → Linear`（2 个矩阵）换成 `Linear → activation × Linear → Linear`（3 个矩阵）。因为多了一倍的输入投影，hidden dim 通常会缩到 2/3 来保持参数量大致相当。
