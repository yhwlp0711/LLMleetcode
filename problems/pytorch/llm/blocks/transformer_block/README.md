# LLaMA-style Transformer Block

把现代 LLM 的核心零件拼装成一个完整的 decoder block。这是**集成题**：考你
能否把前面练过的 RMSNorm / SDPA / SwiGLU FFN 正确组合起来。

## 结构（pre-norm 版本）

```
x
├──> RMSNorm ──> Self-Attention ──┐
│                                  │
│            ┌─────────────────────┘
└──+ (residual) ──> hidden
hidden
├──> RMSNorm ──> SwiGLU FFN ──┐
│                              │
│         ┌────────────────────┘
└──+ (residual) ──> out
```

注意：**pre-norm**（先 norm 再 attention/ffn），是 LLaMA 等现代 LLM 的标
配（vs 原始 Transformer 的 post-norm）。

## 待实现类

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, norm_eps: float = 1e-6):
        ...
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        ...
```

## 子模块（**judge 提供参考实现作为依赖**）

为了让你专注「拼装」，judge 已经提供了下面四个工具，**请直接 import 使
用**，不要重新实现：

```python
from mlleetcode.reference import rms_norm, sdpa, swiglu_ffn_forward
```

- `rms_norm(x, weight, eps)` — RMSNorm 函数式接口
- `sdpa(q, k, v, mask)` — scaled dot-product attention
- `swiglu_ffn_forward(x, gate_w, up_w, down_w)` — LLaMA-style FFN

## `__init__` 参数

按以下名字创建参数/子层（judge 用 `load_state_dict` 同步）：

- `self.attn_norm_weight`: `nn.Parameter(torch.ones(d_model))`  ← attention 前 RMSNorm
- `self.W_q`, `self.W_k`, `self.W_v`, `self.W_o`: 四个 `nn.Parameter(torch.zeros(d_model, d_model))`
  （`nn.Linear` 也行，但为了简化判分，**这里用裸 Parameter**，初值随便）
- `self.ffn_norm_weight`: `nn.Parameter(torch.ones(d_model))`  ← FFN 前 RMSNorm
- `self.gate_proj`, `self.up_proj`: `nn.Parameter(torch.zeros(d_model, d_ff))`
- `self.down_proj`: `nn.Parameter(torch.zeros(d_ff, d_model))`

把 `num_heads`、`d_model`、`norm_eps` 保存为属性。

## `forward` 流程

```python
def forward(self, x, mask=None):
    # 1. Attention sub-block
    h = rms_norm(x, self.attn_norm_weight, self.norm_eps)
    # 投影 + 切多头
    q = (h @ self.W_q).reshape(B, T, num_heads, head_dim).transpose(1, 2)
    k = (h @ self.W_k).reshape(B, T, num_heads, head_dim).transpose(1, 2)
    v = (h @ self.W_v).reshape(B, T, num_heads, head_dim).transpose(1, 2)
    attn_out = sdpa(q, k, v, mask)                                # (B, H, T, d_h)
    attn_out = attn_out.transpose(1, 2).reshape(B, T, D) @ self.W_o
    x = x + attn_out

    # 2. FFN sub-block
    h = rms_norm(x, self.ffn_norm_weight, self.norm_eps)
    ffn_out = swiglu_ffn_forward(h, self.gate_proj, self.up_proj, self.down_proj)
    return x + ffn_out
```

## 简化

为了把题目规模控制住，本题**不**含 RoPE、GQA、KV cache —— 那些都在独立
题里考过。这里专注「block 装配」。

## 判分

- **Init**：参数名 + shape 正确
- **Forward**：reference 权重通过 `load_state_dict` 注入后，对比输出
- 容差 `atol=1e-5`
