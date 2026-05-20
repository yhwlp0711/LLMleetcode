# 激活函数

实现现代 Transformer / LLM 里常用的激活函数。都是纯函数，无参数无状态。

## 待实现函数

### 1. `silu(x)`（也叫 Swish）

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

LLaMA、Mistral、Gemma 都用它。是 ReLU 的平滑近似。

### 2. `gelu_exact(x)`

精确版 GELU：

$$\text{GELU}(x) = x \cdot \Phi(x) = \frac{x}{2}\bigl(1 + \operatorname{erf}(x / \sqrt{2})\bigr)$$

用 `torch.erf`。BERT、GPT-2（不用 tanh 近似时）使用此公式。

### 3. `gelu_tanh(x)`

GELU 的 tanh 近似（原论文版本，GPT-2/3 用它）：

$$\text{GELU}_{\tanh}(x) = \frac{x}{2}\bigl(1 + \tanh\bigl[\sqrt{2/\pi}\,(x + 0.044715\,x^3)\bigr]\bigr)$$

### 4. `swiglu(x, gate)`

LLaMA FFN 里用的「Swish-Gated Linear Unit」。给两个**同 shape**张量：

$$\text{SwiGLU}(x, \text{gate}) = \text{SiLU}(\text{gate}) \odot x$$

**顺序很重要**：SiLU 作用在 `gate` 上，再与 `x` 逐元素相乘。

### 5. `geglu(x, gate)`

GELU-Gated Linear Unit（T5 v1.1、Gemma、PaLM 用）：

$$\text{GeGLU}(x, \text{gate}) = \text{GELU}(\text{gate}) \odot x$$

这里用**精确版** GELU（不是 tanh 近似）。

## 说明

- 输入是 `torch.float32`。**禁止用** `F.gelu` / `F.silu` / `F.glu` 等内置
  函数，自己按公式实现。
- 容差 `atol=1e-6`。
