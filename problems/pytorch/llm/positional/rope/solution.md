# 解题思路：Rotary Position Embeddings (RoPE)

## 一句话思路

旋转位置编码（RoPE）把位置信息编码进 Q/K 向量本身：把相邻两个特征
`(x_{2i}, x_{2i+1})` 看作复平面上的一个点，按位置 $m$ 和频率 $\theta_i$ 决定的
角度做**二维旋转**。旋转后 Q·K 的内积只和**相对位置**有关——这是 RoPE 编码相对
距离的核心机制。

## 从直觉到公式

### 核心直觉：旋转保持距离、只看相对角度

两个向量各自旋转后做内积：$\langle R(m\theta)\,q,\; R(n\theta)\,k \rangle$ 只
依赖角度差 $(m-n)\theta$，不依赖绝对位置 $m$、$n$。所以注意力打分天然反映「两个
token 离多远」，不需要额外的相对位置 bias。

### 频率设计

对第 $i$ 对特征（$i = 0, 1, \dots, d/2 - 1$，$d$ = head_dim）：

$$\theta_i = \frac{1}{\text{base}^{\,2i / d}}$$

低维 $i$ 对应高频率（快速旋转，捕捉短距离），高维 $i$ 对应低频率（慢速旋转，覆
盖长距离）。`base=10000` 是来自原始 Transformer 的经验值。

位置 $m$、频率 $\theta_i$ 的角度就是 $m \cdot \theta_i$。

### 旋转的实现

每对 $(x_{2i}, x_{2i+1})$ 按角度 $\phi = m\theta_i$ 做旋转：

$$\text{out}[\ldots, 2i] = x_{2i} \cos\phi - x_{2i+1} \sin\phi$$

$$\text{out}[\ldots, 2i+1] = x_{2i+1} \cos\phi + x_{2i} \sin\phi$$

向量化的优雅做法：构造一个「负旋转」张量 `x_rotated`，其中每对 $(a, b)$ 变成
$(-b, a)$（绕原点逆时针转 90°），然后：

$$\text{out} = x \cdot \cos + x_{\text{rot}} \cdot \sin$$

一次 elementwise 乘法完成全部旋转。

## 参考实现

### `build_rope_cache`

预计算 cos/sin 查找表，只算一次、反复复用：

```python
import torch

def build_rope_cache(seq_len, head_dim, base=10000.0):
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) * 2 / head_dim))
    pos = torch.arange(seq_len, dtype=torch.float32)
    angles = pos[:, None] * inv_freq[None, :]    # (seq_len, half) 外积
    cos = angles.cos().repeat_interleave(2, dim=-1)  # (seq_len, head_dim)
    sin = angles.sin().repeat_interleave(2, dim=-1)
    return cos, sin
```

### `apply_rope`

```python
def apply_rope(x, cos, sin):
    B, H, T, D = x.shape
    cos_b = cos.view(1, 1, T, D)               # 广播到 batch/head
    sin_b = sin.view(1, 1, T, D)

    x_pairs = x.reshape(B, H, T, D // 2, 2)   # 每对相邻元素分组
    a, b = x_pairs[..., 0], x_pairs[..., 1]
    x_rotated = torch.stack([-b, a], dim=-1).reshape(B, H, T, D)  # (-b, a)

    return x * cos_b + x_rotated * sin_b
```

## 关键点

1. **`repeat_interleave(2, dim=-1)` 让 cos/sin 和 x 同 shape**。每个频率对应一
   对特征 `(2i, 2i+1)`，所以 cos/sin 从 `(T, half)` 扩成 `(T, head_dim)`，相邻
   两位放相同值。这样后续旋转只需 elementwise 乘，不用 gather。注意
   `repeat_interleave` 是 `[a, a, b, b]`，和 `repeat`（`[a, b, a, b]`）不同。

2. **「负旋转」张量的构造**。把 `x` reshape 成 `(..., D//2, 2)`，取出每对 `(a, b)`，
   `stack([-b, a])` 再 reshape 回去，得到 `[-x_1, x_0, -x_3, x_2, ...]`。这等价
   于给每对做 90° 旋转——是 RoPE 公式里的核心几何操作。

3. **位置 0 是恒等映射**。角度 = 0 时 cos=1、sin=0，`out = x·1 + x_rotated·0 = x`。
   判分常用这个 property 作为快速正确性检查。

4. **cos/sin 用 `.view(1, 1, T, D)` 广播**，让同一份表对所有 batch、所有 head 复
   用。RoPE 是无参数（no learnable parameters）的——频率完全由公式决定。

5. **延伸**：RoPE 已是事实标准（LLaMA、Mistral、Qwen、Gemma 都用），因为它可外推
   （推理时直接扩展 cos/sin 表到更长位置）、无需可训练参数、且天然编码相对位置。
   更早的方案是加性的正弦位置编码（见 `pytorch.llm.positional.sinusoidal_pe`），RoPE
   是它的乘性升级。把 RoPE 集成到注意力里就是完整 LLaMA block（见
   `pytorch.llm.blocks.transformer_block`）。
