# 解题思路：Multi-Head Attention（纯函数版）

## 参考实现

```python
from math import sqrt
import torch.nn.functional as F

def mha(x, W_q, W_k, W_v, W_o, num_heads, mask=None):
    B, T, D = x.shape
    head_dim = D // num_heads

    # 1. Q/K/V 投影
    q = x @ W_q                                       # (B, T, D)
    k = x @ W_k
    v = x @ W_v

    # 2. 切分多头：(B, T, D) → (B, T, H, head_dim) → (B, H, T, head_dim)
    def split(t):
        return t.reshape(B, T, num_heads, head_dim).transpose(1, 2)
    q, k, v = split(q), split(k), split(v)

    # 3. SDPA per head
    scores = q @ k.transpose(-2, -1) / sqrt(head_dim)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    out = attn @ v                                    # (B, H, T, head_dim)

    # 4. 合并多头：(B, H, T, head_dim) → (B, T, H, head_dim) → (B, T, D)
    out = out.transpose(1, 2).reshape(B, T, D)

    # 5. 输出投影
    return out @ W_o
```

## 形状变换图

```
x:               (B, T, D)
  └─ @ W_q ────→ q: (B, T, D)
      └─ reshape (B, T, H, d_h)
          └─ transpose(1,2) → q': (B, H, T, d_h)
                  ↓
              [SDPA]
                  ↓
          out: (B, H, T, d_h)
          └─ transpose(1,2) → (B, T, H, d_h)
              └─ reshape (B, T, D)
                  └─ @ W_o → 输出: (B, T, D)
```

## 关键技巧

### 1. 切分多头：`reshape` + `transpose` 的顺序

```python
t.reshape(B, T, H, d_h).transpose(1, 2)
# (B, T, D) → (B, T, H, d_h) → (B, H, T, d_h)
```

**为什么要先 reshape 再 transpose？** reshape 不改变内存布局，只重新解释；
transpose 改变 stride，让最后两维变成 `(T, d_h)`，正好是 SDPA 期望的输
入。

**反过来不行**：先 `transpose(1, 2)` 把 `(B, T, D)` 变成 `(B, D, T)`，再
reshape 出来就是错的（每行的元素被打散了）。这是新手常踩的坑。

### 2. 合并多头：必须先 transpose 再 reshape

```python
out = out.transpose(1, 2).reshape(B, T, D)
```

直接 `out.reshape(B, T, D)` 会把多个 head 的特征**沿 token 维度交错**起
来，完全错误。一定要先 transpose 把 H 和 T 换回来。

**踩坑警示**：`transpose` 后的张量是 non-contiguous，`.view()` 会报错。
用 `.reshape()`（PyTorch 会自动 `.contiguous().view()`）。

### 3. `sqrt(head_dim)` 不是 `sqrt(D)`

SDPA 里的缩放因子是「key 维度」，单头是 `D`，多头时每头的 key 维度是
`head_dim = D / H`。**用错了会导致 attention 分布偏离最优**，虽然形状对
但数值差。

### 4. mask 的 shape

mask 要能广播到 `(B, H, T, T)`。题目用 `(B, 1, T, T)` —— 第二维设为 1，
让所有 head 共享同一份 mask（典型的 causal / padding mask 用法）。

## 为什么要切分多头？

单头注意力的「注意力分布」只有一组，模型只能捕捉单一类型的相关性
（比如「指代消歧」一种关系）。多头让模型并行学多组不同的「关注模式」：

- Head 1 可能关注语法依赖
- Head 2 可能关注共指
- Head 3 可能关注主题
- ...

每个 head 维度小（`d_h = D/H`），所以总参数量不变；但表达力大大增强。

## Pattern A vs Pattern B

本题是 Pattern A（纯函数，权重外部给）。Pattern B 版本（`class MHA(nn.Module)`，
自己定义 `q_proj`、`k_proj` 等 `nn.Linear`）暂时没出，留作进阶。两者算法
一模一样，只是工程封装不同。
