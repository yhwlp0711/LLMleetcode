# 解题思路：Multi-Head Attention（纯函数版）

## 一句话思路

多头注意力（multi-head attention, MHA）就是把输入投影成 Q/K/V 后**切成好几个头
（head）并行做注意力**，再把各头结果拼回来过一次输出投影。核心是「投影 → 切头
→ 每头做 SDPA → 合头 → 输出投影」这条流水线，难点全在 reshape/transpose 的顺序
别搞错。

## 拆解思路

### 为什么要切多头？

单头注意力只有一组「关注模式」，模型只能捕捉单一类型的相关性。多头让模型并行学
多组不同的关注方式——比如一个头盯语法依赖、一个头盯共指、一个头盯主题。每个头维
度小（`head_dim = D / num_heads`），所以总计算量不变，但表达力大大增强。

### 五步流水线

1. **投影**：`q = x @ W_q`，K、V 同理，形状都是 `(B, T, D)`。
2. **切头**：把 `D` 维拆成 `num_heads × head_dim`，reshape 到
   `(B, T, num_heads, head_dim)`，再 transpose 到 `(B, num_heads, T, head_dim)`，
   让「头」维排到前面，每个头就能独立算注意力。
3. **每头做 SDPA**（见 `pytorch.llm.attention.scaled_dot_product_attention`）：

$$\text{softmax}\!\left(\frac{QK^\top}{\sqrt{\text{head\_dim}}} + \text{Mask}\right)V$$

4. **合头**：transpose 回 `(B, T, num_heads, head_dim)`，再 reshape 成 `(B, T, D)`。
5. **输出投影**：`out @ W_o`，形状 `(B, T, D)`。

## 参考实现

```python
import torch.nn.functional as F
from math import sqrt

def mha(x, W_q, W_k, W_v, W_o, num_heads, mask=None):
    B, T, D = x.shape
    head_dim = D // num_heads

    q, k, v = x @ W_q, x @ W_k, x @ W_v          # 投影，都是 (B, T, D)

    def split(t):                                 # (B,T,D) → (B,H,T,head_dim)
        return t.reshape(B, T, num_heads, head_dim).transpose(1, 2)
    q, k, v = split(q), split(k), split(v)

    scores = q @ k.transpose(-2, -1) / sqrt(head_dim)  # 缩放用 head_dim
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    out = attn @ v                                # (B, H, T, head_dim)

    out = out.transpose(1, 2).reshape(B, T, D)    # 先 transpose 再 reshape
    return out @ W_o
```

## 关键点

1. **切头：先 reshape 再 transpose，顺序不能反**。reshape 只是重新解释同一块内
   存，把 `D` 拆成 `(num_heads, head_dim)`；transpose 再把「头」维换到前面。如果
   先 transpose 成 `(B, D, T)` 再 reshape，每行的元素就被打散了，结果全错。

2. **合头：必须先 transpose 再 reshape**。直接 `out.reshape(B, T, D)` 会把不同
   头的特征沿 token 维交错在一起。要先 `transpose(1, 2)` 把头维换回 `T` 后面。另
   外 transpose 后的张量在内存里不连续（non-contiguous），`.view()` 会报错，用
   `.reshape()`（它会自动处理连续性）。

3. **缩放因子是 `sqrt(head_dim)`，不是 `sqrt(D)`**。每个头独立做注意力，key 维
   度是 `head_dim = D / num_heads`。用错了形状对但数值偏，attention 分布会失准。

4. **mask 广播到 `(B, num_heads, T, T)`**。题目常给 `(B, 1, T, T)`，第二维为 1
   让所有头共享同一份 mask（典型 causal / padding 用法，见
   `pytorch.llm.attention.causal_mask`）。`masked_fill(~mask, -inf)` 把 False 位
   置屏蔽。

5. **延伸**：MHA 里 Q/K/V 头数相同（1:1）。让多个 Q 头共享少量 K/V 头能省显存，
   这就是 `pytorch.llm.attention.gqa`；把它拼进完整 decoder 层见
   `pytorch.llm.blocks.transformer_block`。
