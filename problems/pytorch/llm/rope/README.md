# Rotary Position Embeddings (RoPE)

实现 RoPE（Su 等人，2021），LLaMA、GPT-NeoX 与几乎所有现代 decoder-only
LLM 的位置编码方案。RoPE 通过**旋转**特征对来编码 token 的位置。

## 待实现函数

### 1. `build_rope_cache(seq_len, head_dim, base=10000.0)`
预计算用于旋转 Q 和 K 的 cos/sin 表。

```python
def build_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    返回:
        cos: (seq_len, head_dim) 的 cos 值
        sin: (seq_len, head_dim) 的 sin 值
    """
```

对每对特征索引 `(2i, 2i+1)`（`i = 0, 1, ..., head_dim/2 - 1`）和位置 `m`：

- 频率：$\theta_i = 1 / \text{base}^{2i / \text{head\_dim}}$
- 角度：$m \cdot \theta_i$

cos/sin 表里 `cos[m, 2i] = cos[m, 2i+1] = cos(m * θ_i)`（成对复制相同值），
sin 同理。这样旋转就是一个 elementwise 乘法。

`head_dim` 保证是偶数。

### 2. `apply_rope(x, cos, sin)`
把 RoPE 应用到 shape `(B, H, T, head_dim)` 的张量上：

```python
def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    参数:
        x:   (B, H, T, head_dim)
        cos: (T, head_dim)   来自 build_rope_cache(T, head_dim)
        sin: (T, head_dim)
    返回:
        与 x 形状相同的旋转后张量。
    """
```

对每对 `(x[..., 2i], x[..., 2i+1])` 做旋转：

$$\begin{aligned}
\text{out}[..., 2i]   &= x[..., 2i]   \cdot \cos(m \theta_i) - x[..., 2i+1] \cdot \sin(m \theta_i) \\
\text{out}[..., 2i+1] &= x[..., 2i+1] \cdot \cos(m \theta_i) + x[..., 2i]   \cdot \sin(m \theta_i)
\end{aligned}$$

**优雅的形式**：把 `x` 的每对相邻元素「交换并把第一个取负」，得到
`x_rotated = (-x_2, x_1, -x_4, x_3, ...)`。则
`out = x * cos + x_rotated * sin`。

## 说明

- 输入是 `torch.float32`。
- 容差 `atol=1e-5`。
