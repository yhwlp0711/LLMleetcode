# 解题思路：Rotary Position Embeddings (RoPE)

## 直觉先行

RoPE 的核心想法：**把位置信息编码进 Q/K 向量本身**，做法是「按位置旋转
特征对」。两个相邻特征 `(x_{2i}, x_{2i+1})` 看作复平面上的点，按角度
$m \theta_i$ 旋转，旋转矩阵：

$$R(\phi) = \begin{pmatrix}\cos\phi & -\sin\phi \\ \sin\phi & \cos\phi\end{pmatrix}$$

旋转后的内积有个绝妙性质：$\langle R(m\theta) q, R(n\theta) k\rangle$ 只与
**相对位置** $m - n$ 有关。这就是 RoPE 让模型理解「相对距离」的核心机制。

## 1. `build_rope_cache`

```python
def build_rope_cache(seq_len, head_dim, base=10000.0):
    half = head_dim // 2
    # theta_i = 1 / base^(2i / head_dim)，i 取 [0, half)
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) * 2 / head_dim))
    pos = torch.arange(seq_len, dtype=torch.float32)
    angles = pos[:, None] * inv_freq[None, :]    # (seq_len, half)
    cos = angles.cos()
    sin = angles.sin()
    # 成对复制：(seq_len, half) → (seq_len, head_dim)
    cos = cos.repeat_interleave(2, dim=-1)
    sin = sin.repeat_interleave(2, dim=-1)
    return cos, sin
```

### 关键步骤拆解

1. **频率序列** `inv_freq`：`head_dim/2` 个递减频率。低维特征旋转慢（编码
   长距离），高维特征旋转快（编码短距离）。`base=10000` 是 Transformer 论
   文里 sinusoidal PE 的传统。
2. **角度矩阵** `pos[:, None] * inv_freq[None, :]`：外积广播成 `(T, half)`，
   `angles[m, i] = m * theta_i`。
3. **`repeat_interleave(2, dim=-1)`**：把每个元素**重复 2 次**变成相邻的
   一对。例如 `[a, b, c]` 变成 `[a, a, b, b, c, c]`。这样后续 `apply_rope`
   直接用 elementwise 乘就行，不用每对单独处理。

> ⚠️ `repeat_interleave(2)` ≠ `repeat(2)`：后者是 `[a, b, c, a, b, c]`。

## 2. `apply_rope`

```python
def apply_rope(x, cos, sin):
    B, H, T, D = x.shape
    cos_b = cos.view(1, 1, T, D)    # 广播到 batch / head
    sin_b = sin.view(1, 1, T, D)

    # 把 x 按相邻两两分组，每对 (a, b) → (-b, a)
    x_pairs = x.reshape(B, H, T, D // 2, 2)
    a = x_pairs[..., 0]
    b = x_pairs[..., 1]
    x_rotated = torch.stack([-b, a], dim=-1).reshape(B, H, T, D)

    return x * cos_b + x_rotated * sin_b
```

### 「负旋转」张量怎么构造？

希望得到：

```
原: [x_0, x_1, x_2, x_3, x_4, x_5, ...]
负旋: [-x_1, x_0, -x_3, x_2, -x_5, x_4, ...]
```

代码做法：

1. `reshape` 成 `(..., D/2, 2)`，把相邻两元素归到最后一维。
2. 取 `[..., 0]` 和 `[..., 1]` 分别是每对的第一个、第二个。
3. `torch.stack([-b, a], dim=-1)`：把 `-b` 放在偶数位、`a` 放在奇数位。
4. `reshape` 回 `(..., D)` 展平。

数学上等价于「每对绕原点逆时针转 90°」，这是 RoPE 公式里的核心几何操作。

### 验证：位置 0 是恒等映射

位置 `m=0` 时角度全为 0，`cos=1, sin=0`，所以 `out = x * 1 + x_rotated * 0
= x`。这是题目里的「property 测试用例」检查的。

## 为什么要成对复制 cos/sin？

如果不复制，cos/sin 是 `(T, D/2)`，每个 cos 值要应用到一对 `(x_{2i},
x_{2i+1})` 上 —— 需要 `gather` / `unsqueeze` 来铺开。复制后 cos/sin 是
`(T, D)`，跟 `x` 同 shape，直接 elementwise 乘就行，代码极简。

**内存代价**：cos/sin 大了一倍。但 RoPE cache 通常只算一次缓存复用，无所
谓。

## RoPE 为什么这么受欢迎？

1. **可外推**：训练时见过的最大位置是 `T`，推理时可以直接用更大的 `T'`，
   只要扩展 cos/sin 表。绝对位置 embedding 做不到。
2. **相对位置自然涌现**：$qk^\top$ 在 RoPE 下变成相对位置的函数，无需额
   外加 bias。
3. **无参数**：纯函数，不增加可训练参数。

LLaMA、Mistral、Qwen、Gemma 全用 RoPE，已是事实标准。
