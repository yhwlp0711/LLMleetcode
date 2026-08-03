# 解题思路：LLaMA-style Transformer Block

## 这道题考什么

不是考算法（前面 10 道已经分别考过）—— 考你**会不会把零件正确地拼成一
个 block**。这是工程能力测试，重点：

1. **pre-norm 顺序**：先 norm 再 attention，不是反过来
2. **两次 residual**：attention 和 FFN 各一次
3. **多头切分维度**：Q/K/V 都按 `(B, T, num_heads, head_dim)` 切
4. **参数命名**：跟 judge 约定一致，让 `load_state_dict` 注入成功

## 参考实现

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, norm_eps=1e-6):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.norm_eps = norm_eps

        self.attn_norm_weight = nn.Parameter(torch.ones(d_model))
        self.W_q = nn.Parameter(torch.zeros(d_model, d_model))
        self.W_k = nn.Parameter(torch.zeros(d_model, d_model))
        self.W_v = nn.Parameter(torch.zeros(d_model, d_model))
        self.W_o = nn.Parameter(torch.zeros(d_model, d_model))

        self.ffn_norm_weight = nn.Parameter(torch.ones(d_model))
        self.gate_proj = nn.Parameter(torch.zeros(d_model, d_ff))
        self.up_proj = nn.Parameter(torch.zeros(d_model, d_ff))
        self.down_proj = nn.Parameter(torch.zeros(d_ff, d_model))

    def forward(self, x, mask=None):
        B, T, D = x.shape

        # ---- Self-Attention sub-block ----
        h = rms_norm(x, self.attn_norm_weight, self.norm_eps)
        q = (h @ self.W_q).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = (h @ self.W_k).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = (h @ self.W_v).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        attn_out = sdpa(q, k, v, mask)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, D) @ self.W_o
        x = x + attn_out

        # ---- FFN sub-block ----
        h = rms_norm(x, self.ffn_norm_weight, self.norm_eps)
        ffn_out = swiglu_ffn_forward(h, self.gate_proj, self.up_proj, self.down_proj)
        return x + ffn_out
```

## Pre-norm vs Post-norm

```
Post-norm（原版 Transformer 2017）:
    x → attn → +residual → norm → ffn → +residual → norm

Pre-norm（GPT-2, LLaMA, 现代标配）:
    x → norm → attn → +residual → norm → ffn → +residual
```

为什么 pre-norm 胜出？训练**稳定性**：
- post-norm 在深层堆叠时容易发散
- pre-norm 让 residual stream「直通」，梯度更直接
- 代价是最终输出没归一化（通常在 block 之后再加一个 final norm）

## 「Residual stream」直觉

把 `x` 想象成一条「主干道」，每个 sub-block 像一个出口：

```
x ───────────────────────────────────────> x'
           │                  │
           ↓                  ↓
         attn               FFN
           │                  │
           └──+ ───→  ←───────┘
              ↑
            (residual add 回主干)
```

`x` 一路保留原始信息；attention 和 FFN 各加一份「修正」。这是为什么深层
Transformer 能训得起来的关键。

## 关键点

### 1. 多头切分用 `.reshape(B, T, H, d_h).transpose(1, 2)`

不能直接 `.reshape(B, H, T, d_h)` —— 数据会被打乱（特征跨头交错）。正
确顺序：先变成 `(B, T, H, d_h)`（把 D 切成 H × d_h），再交换 H 和 T 两
个轴。

### 2. 合头要 `.transpose(1, 2).reshape(B, T, D)`

跟切头反过来：先 transpose 还原 `(B, T, H, d_h)`，再 reshape 拼回
`(B, T, D)`。**直接 `.reshape(B, T, D)` 会错**，因为 transpose 之后的
张量 stride 不连续。`.reshape()` 会先 `.contiguous()` 再 reshape，
PyTorch 帮我们处理。

### 3. mask 传给 sdpa

`mask` 一路从 `forward` 参数传到 `sdpa` 调用。判分会传 causal mask，shape
`(1, 1, T, T)`，sdpa 内部会广播到 `(B, num_heads, T, T)`。

### 4. 两个 norm 参数**独立**

`attn_norm_weight` 和 `ffn_norm_weight` 是两个不同的可学习参数 —— 不要
共享。两个 RMSNorm 在不同的位置看到不同的统计分布，独立学习。

## 为什么 judge 提供 `mlleetcode.reference`

这是 Pattern A → Pattern B 的桥梁设计：

- **如果不提供**，本题就要求用户重新实现 RMSNorm / SDPA / SwiGLU FFN
  —— 这变成了 4 道题混杂，错一处就全错。
- **提供后**，用户专注「block 装配」考点，前面的零件 bug 不会污染这道
  题。

这种「**集成测试用经过验证的零件，单元测试单独考零件**」是工程项目里
的常规做法。

## 简化点

真实的 LLaMA Block 还包含：

- **RoPE**：应用到 Q/K 上
- **GQA**：K/V 头数少于 Q
- **KV cache**：推理时的状态管理

本题为了控制时间和复杂度都省了。如果你已经做过 `rope` / `gqa` /
`kv_cache` 三道题，组合起来就是工业级 LLaMA Block。

## 「为什么不写 nn.Linear？」

本题用裸 `nn.Parameter` 而不是 `nn.Linear`，是为了：

1. 跟 `sdpa` / `swiglu_ffn_forward` 这种函数式接口对齐 —— 它们都接 raw
   weight tensor
2. 让 judge 的 `assert_param_names` 检查更直接（不用嵌套 `xxx.weight`）

工业代码当然用 `nn.Linear`（更优雅、更易复用）。这道题简化为参数张量
是判分友好的选择。
