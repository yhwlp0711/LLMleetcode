# 解题思路：门控激活函数（SwiGLU / GeGLU）

## 一句话思路

门控激活（gated activation）就是「用一个激活函数处理 `gate`，再逐元素乘上 `x`」。
本题实现两个门控线性单元（Gated Linear Unit, GLU）变体：SwiGLU 用 SiLU 当门，
GeGLU 用 GELU 当门。**顺序是关键**——激活套在 `gate` 上，不是 `x` 上。

## 从直觉到公式

### 「门控」是什么意思

普通激活只有一个输入：`activation(z)`。门控激活有两个同 shape 的输入，一个当「值」，
一个被激活后当「门」，控制让多少值通过：

$$\text{GLU}(x, \text{gate}) = \text{activation}(\text{gate}) \odot x$$

$\odot$ 是逐元素相乘。直觉上，模型可以对每个位置学一个「开多大」的门，形成一种条件
性、乘法式的信息通路，表达能力比单条加法通路更强。

### 两个变体

SwiGLU（LLaMA / Mistral 前馈网络用）用 SiLU 当门：

$$\text{SwiGLU}(x, \text{gate}) = \text{SiLU}(\text{gate}) \odot x, \qquad \text{SiLU}(z) = z \cdot \sigma(z)$$

GeGLU（T5 v1.1 / Gemma / PaLM 用）用精确版 GELU 当门：

$$\text{GeGLU}(x, \text{gate}) = \text{GELU}(\text{gate}) \odot x, \qquad \text{GELU}(z) = \frac{z}{2}\bigl(1 + \operatorname{erf}(z/\sqrt{2})\bigr)$$

## 参考实现

激活部分内联写就行（`torch.sigmoid` / `torch.erf` 允许用，被禁的是 `F.silu` /
`F.gelu` 这类现成函数）：

```python
from math import sqrt

def swiglu(x, gate):
    silu_gate = gate * torch.sigmoid(gate)          # SiLU 作用在 gate 上
    return silu_gate * x                            # 再逐元素乘 x

def geglu(x, gate):
    gelu_gate = 0.5 * gate * (1.0 + torch.erf(gate / sqrt(2.0)))  # 精确版 GELU
    return gelu_gate * x
```

## 关键点

1. **顺序不能搞反**：激活函数套在 `gate` 上，`x` 保持原样直接相乘。写成
   `silu(x) * gate` 就错了。真实的 LLaMA 前馈网络里，`gate` 对应
   `gate_proj(x)`、`x` 对应 `up_proj(x)`；本题只考门控这一步运算，不涉及投影矩阵。

2. **GeGLU 用精确版 GELU**：这里要用 `torch.erf` 的精确公式，不是 tanh 近似。两者数
   值有微小差异，判分对齐的是精确版。

3. **为什么现代 LLM 爱用门控前馈网络**：经典前馈是
   `Linear → 激活 → Linear`（2 个矩阵），门控版是
   `Linear → 激活 × Linear → Linear`（3 个矩阵）。多了一路输入投影，容量更强；为了
   保持总参数量相当，隐藏维度通常缩到约 2/3。

4. **延伸**：单独的 SiLU / GELU（不带门控）见 `pytorch.nn.activations`；作为门的核心
   之一的 sigmoid，其数值稳定实现见 `pytorch.nn.numeric_activations`。
