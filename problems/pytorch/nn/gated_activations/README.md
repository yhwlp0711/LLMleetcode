# 门控激活函数（SwiGLU / GeGLU）

现代 LLM 的 FFN 大量使用「门控线性单元」（Gated Linear Unit, GLU）变体。
给两个**同 shape** 的张量 `x` 和 `gate`，用一个激活函数作用在 `gate` 上，再与 `x` 逐元素相乘。

本题实现两个 GLU 变体。

## 待实现函数

### 1. `swiglu(x, gate)`

LLaMA / Mistral FFN 用的「Swish-Gated Linear Unit」：

$$\text{SwiGLU}(x, \text{gate}) = \text{SiLU}(\text{gate}) \odot x, \qquad \text{SiLU}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}$$

**顺序很重要**：SiLU 作用在 `gate` 上，再与 `x` 逐元素相乘。

### 2. `geglu(x, gate)`

GELU-Gated Linear Unit（T5 v1.1、Gemma、PaLM 用）：

$$\text{GeGLU}(x, \text{gate}) = \text{GELU}(\text{gate}) \odot x, \qquad \text{GELU}(z) = \frac{z}{2}\bigl(1 + \operatorname{erf}(z / \sqrt{2})\bigr)$$

这里用**精确版** GELU（`torch.erf`，不是 tanh 近似）。

## 说明

- 输入是 `torch.float32`，`x` 与 `gate` 同 shape。
- **禁止用** `F.silu` / `F.gelu` / `F.glu` 等内置门控/激活函数，自己按公式实现（激活部分内联在 `swiglu` / `geglu` 里即可）。
- 单独的 SiLU / GELU 见 `pytorch.nn.activations`。
- 容差 `atol=1e-6`。
