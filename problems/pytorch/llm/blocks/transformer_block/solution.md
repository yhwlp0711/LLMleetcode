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

judge 已经提供 `rms_norm`、`sdpa`、`swiglu_ffn_forward` 三个工具，直接 import 用，
不用重写：

```python
import torch
import torch.nn as nn
from mlleetcode.reference import rms_norm, sdpa, swiglu_ffn_forward

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, norm_eps=1e-6):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model, self.num_heads = d_model, num_heads
        self.head_dim = d_model // num_heads
        self.norm_eps = norm_eps

        self.attn_norm_weight = nn.Parameter(torch.ones(d_model))
        self.W_q = nn.Parameter(torch.zeros(d_model, d_model))
        self.W_k = nn.Parameter(torch.zeros(d_model, d_model))
        self.W_v = nn.Parameter(torch.zeros(d_model, d_model))
        self.W_o = nn.Parameter(torch.zeros(d_model, d_model))

        self.ffn_norm_weight = nn.Parameter(torch.ones(d_model))
        self.gate_proj = nn.Parameter(torch.zeros(d_model, d_ff))
        self.up_proj   = nn.Parameter(torch.zeros(d_model, d_ff))
        self.down_proj = nn.Parameter(torch.zeros(d_ff, d_model))

    def forward(self, x, mask=None):
        B, T, D = x.shape

        # 1. Attention 子块（pre-norm）
        h = rms_norm(x, self.attn_norm_weight, self.norm_eps)
        q = (h @ self.W_q).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = (h @ self.W_k).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = (h @ self.W_v).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        attn_out = sdpa(q, k, v, mask)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, D) @ self.W_o
        x = x + attn_out                                      # 残差

        # 2. FFN 子块（pre-norm）
        h = rms_norm(x, self.ffn_norm_weight, self.norm_eps)
        ffn_out = swiglu_ffn_forward(h, self.gate_proj, self.up_proj, self.down_proj)
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

5. **参数名要和 judge 约定一致**。用裸 `nn.Parameter`（而非 `nn.Linear`）是为了对
   齐函数式的 `sdpa` / `swiglu_ffn_forward` 接口，也让 `load_state_dict` 注入参考
   权重更直接。

6. **延伸**：本题为控规模省掉了 RoPE、GQA、KV cache。把
   `pytorch.llm.positional.rope`（旋转位置编码）、`pytorch.llm.attention.gqa`、
   `pytorch.llm.attention.kv_cache` 三块补进来，就是工业级的 LLaMA block。
