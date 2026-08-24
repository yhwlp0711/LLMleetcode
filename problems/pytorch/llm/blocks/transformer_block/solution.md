# 解题思路：LLaMA-style Transformer Block

## 一句话思路

这是一道「装配题」：把前面练过的 RMSNorm、注意力（SDPA）、SwiGLU FFN 按 **pre-norm
（前置归一化）** 的方式拼成一个 decoder block。核心是记住结构——两个子块，每个都
是「先 norm → 算 → 加残差（residual）」，别把顺序搞反。

## 拆解思路

### 结构：两个子块 + 两次残差

```
x ──┬─────────────────────────────────┐
    └─ RMSNorm ─ Self-Attention ──(+)──┤  → hidden
                                       │
hidden ─┬──────────────────────────────┐
        └─ RMSNorm ─ SwiGLU FFN ──(+)──┤  → out
```

两个子块结构一样：先对输入做 RMSNorm，再送进 attention / FFN，输出**加回**原始输
入（残差）。attention 子块的残差是 `x`，FFN 子块的残差是 attention 之后的结果。

### 为什么是 pre-norm？

`x → norm → 子块 → +残差`（pre-norm）比原始 Transformer 的
`x → 子块 → +残差 → norm`（post-norm）训练更稳定：残差主干「直通」不被归一化打断，
深层堆叠时梯度更顺畅，不容易发散。GPT-2、LLaMA 等现代 LLM 都用 pre-norm。

### attention 子块内部

RMSNorm 之后，把 `h` 投影成 Q/K/V，各自切成多头 `(B, num_heads, T, head_dim)`，调
用 judge 提供的 `sdpa`，再合头、过 `W_o`，最后加残差。切头/合头的 reshape 顺序和
`pytorch.llm.attention.mha` 完全一样。

## 参考实现

judge 已经提供 `rms_norm`、`sdpa` 两个工具，直接 import 用，不用重写；SwiGLU
FFN 用几个 `nn.Linear` 自己在 `forward` 里拼：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from mlleetcode.reference import rms_norm, sdpa

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, norm_eps=1e-6):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model, self.num_heads = d_model, num_heads
        self.head_dim = d_model // num_heads
        self.norm_eps = norm_eps

        self.attn_norm_weight = nn.Parameter(torch.ones(d_model))
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.ffn_norm_weight = nn.Parameter(torch.ones(d_model))
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj   = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x, mask=None):
        B, T, D = x.shape

        # 1. Attention 子块（pre-norm）
        h = rms_norm(x, self.attn_norm_weight, self.norm_eps)
        q = self.W_q(h).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(h).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(h).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        attn_out = sdpa(q, k, v, mask)
        attn_out = self.W_o(attn_out.transpose(1, 2).reshape(B, T, D))
        x = x + attn_out                                      # 残差

        # 2. FFN 子块（pre-norm，SwiGLU）
        h = rms_norm(x, self.ffn_norm_weight, self.norm_eps)
        ffn_out = self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
        return x + ffn_out                                    # 残差
```

## 关键点

1. **pre-norm：先 norm 再算**。两个子块都是「norm → 子模块 → 加残差」，别写成
   post-norm。这是现代 LLM 训练稳定的关键。

2. **两次残差各自的加数不同**。attention 子块加的是原始 `x`，FFN 子块加的是
   attention 更新后的 `x`。中间那次 `x = x + attn_out` 会更新 `x`，不能漏。

3. **两个 RMSNorm 参数独立**。`attn_norm_weight` 和 `ffn_norm_weight` 是两个不同
   的可学习参数，它们在不同位置看到不同的分布，不要共享。

4. **切头/合头顺序**：切头 `reshape(B,T,H,head_dim).transpose(1,2)`，合头
   `transpose(1,2).reshape(B,T,D)`，跟 `pytorch.llm.attention.mha` 一致。合头必须
   先 transpose 再 reshape（否则头间特征交错），用 `.reshape` 而非 `.view`（转置
   后张量不连续）。

5. **参数名要和 judge 约定一致**。四个注意力投影和三个 FFN 投影都用无 bias 的
   `nn.Linear`（参数名形如 `W_q.weight`），RMSNorm 的缩放向量用 `nn.Parameter`；
   `load_state_dict` 靠这些名字注入参考权重，写错名字会同步失败。

6. **延伸**：本题为控规模省掉了 RoPE、GQA、KV cache。把
   `pytorch.llm.positional.rope`（旋转位置编码）、`pytorch.llm.attention.gqa`、
   `pytorch.llm.attention.kv_cache` 三块补进来，就是工业级的 LLaMA block。
