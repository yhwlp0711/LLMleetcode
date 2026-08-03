# 解题思路：Sinusoidal Position Encoding

## 一句话思路

正弦位置编码（sinusoidal positional encoding, PE）是 Transformer 原始论文的位置编
码方案：给每个位置、每个维度分配一个固定的 sin/cos 值，偶数维放 sin、奇数维放
cos，不同维度用不同频率。核心就是一次「位置 × 频率」的外积广播，再把 sin/cos 交错
填回去。

## 从直觉到公式

### 为什么要位置编码？

注意力本身是「位置无关」的——把输入 token 打乱顺序，attention 输出只是对应打乱，
不知道谁在前谁在后。位置编码给每个位置一个独特的「身份指纹」加到 embedding 上，
让模型能感知顺序。

### 公式

对位置 $pos$ 和维度对 $i$（$i = 0, 1, \dots, d_{\text{model}}/2 - 1$）：

$$\text{PE}[pos, 2i] = \sin\bigl(pos \cdot \theta_i\bigr)$$

$$\text{PE}[pos, 2i{+}1] = \cos\bigl(pos \cdot \theta_i\bigr)$$

其中频率 $\theta_i = 1 / 10000^{2i / d_{\text{model}}}$。低维度的频率高（变化快），
高维度的频率低（变化慢、周期长），形成一组不同尺度的「时钟」，让模型既能感知相
邻 token 的差异，也能分辨远处的位置。

### 实现思路

1. 算出所有频率：`inv_freq = 1 / 10000^(2i / d_model)`，共 $d_{\text{model}}/2$ 个。
2. 构造角度矩阵：`angles[m, i] = m * inv_freq[i]`——就是位置向量和频率向量的外积广
   播（broadcasting），`(seq_len, 1) * (1, d_model/2) → (seq_len, d_model/2)`。
3. 偶数列填 `sin(angles)`，奇数列填 `cos(angles)`。

## 参考实现

```python
import torch

def build_sinusoidal_pe(seq_len, d_model):
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, d_model, 2, dtype=torch.float32) / d_model))
    pos = torch.arange(seq_len, dtype=torch.float32)
    angles = pos[:, None] * inv_freq[None, :]       # (seq_len, d_model/2)

    pe = torch.zeros(seq_len, d_model, dtype=torch.float32)
    pe[:, 0::2] = angles.sin()                      # 偶数维放 sin
    pe[:, 1::2] = angles.cos()                      # 奇数维放 cos
    return pe
```

## 关键点

1. **`torch.arange(0, d_model, 2)` 取偶数索引得到频率**。步长为 2，得到
   `[0, 2, 4, ..., d_model-2]`，共 $d_{\text{model}}/2$ 个，每个对应一个独立频率
   $\theta_i$。

2. **外积广播构造角度矩阵**。`pos[:, None]` 是 `(seq_len, 1)`，`inv_freq[None, :]`
   是 `(1, d_model/2)`，相乘广播到 `(seq_len, d_model/2)`。一次矩阵乘法（或广播乘
   法）算出所有 `(位置, 频率)` 组合的角度。

3. **切片赋值 `[:, 0::2]` 和 `[:, 1::2]`**。`0::2` 取偶数列、`1::2` 取奇数列。
   shape 都是 `(seq_len, d_model/2)`，和 `angles.sin()` 正好对齐，直接赋值。

4. **位置 0 的 PE 是 `[0, 1, 0, 1, ...]`**。`pos=0` 时所有角度为 0，`sin(0)=0`、
   `cos(0)=1`。这是一个有用的快速正确性检查。

5. **延伸**：正弦 PE 是加性的（直接加到 embedding 上），而 RoPE（见
   `pytorch.llm.positional.rope`）是乘性的（旋转 Q/K 向量）。两者共享「不同维度用
   不同频率」的核心设计，但 RoPE 能更显式地编码相对位置、外推性也更好，已成为
   LLaMA、Mistral 等现代 LLM 的标配。
