# 解题思路：因果 + Padding Mask

## 一句话思路

这题是把两个「保留条件」做**交集**：padding 上要求 query 和 key 都是真实 token，
causal（因果）上要求 key 的位置不超过 query。全程用广播（broadcasting）+ 布尔运
算就能一次算出，不用任何循环。

## 拆解思路

### 目标：`out[b, 0, i, j]` 什么时候为 True？

同时满足三条：query 位置 `i` 不是 padding、key 位置 `j` 不是 padding、且（若
causal）`j <= i`。把它拆成两块独立的「保留集」再取交集：

- **padding 保留集**：两端都是真实 token。
- **causal 保留集**：`j <= i`，即一个下三角矩阵。

### padding：用广播做「布尔外积」

`pad_mask` 形状 `(B, T)`。把它一次变成 query 方向、一次变成 key 方向再相与：

```
q_keep = pad_mask[:, :, None]   # (B, T, 1)  对应 query 位置 i
k_keep = pad_mask[:, None, :]   # (B, 1, T)  对应 key   位置 j
base   = q_keep & k_keep        # (B, T, T)  两端都真实才 True
```

这就是布尔版的「外积」：`base[b, i, j] = pad[b, i] & pad[b, j]`。因为 query 那一
维也参与了，padding 位置对应的**整行**会被清空——正好满足题目要求。

### causal：下三角矩阵

`torch.tril(torch.ones(T, T, bool))` 生成下三角（含主对角线）的 True 矩阵，第 `i`
行只有 `j <= i` 为 True。和 `base` 相与即可。

### 补上 head 维

输出要 `(B, 1, T, T)`，中间的 `1` 是 head 维占位，`unsqueeze(1)` 加上即可。这样
后续 SDPA（见 `pytorch.llm.attention.scaled_dot_product_attention`）能把它广播到
`(B, num_heads, T, T)`。

## 参考实现

```python
import torch

def build_attention_mask(pad_mask, causal):
    B, T = pad_mask.shape
    q_keep = pad_mask[:, :, None]                # (B, T, 1)
    k_keep = pad_mask[:, None, :]                # (B, 1, T)
    base = q_keep & k_keep                       # (B, T, T) 两端都非 padding

    if causal:
        tri = torch.tril(torch.ones(T, T, dtype=torch.bool, device=pad_mask.device))
        base = base & tri                        # 叠加 j <= i

    return base.unsqueeze(1)                      # (B, 1, T, T)
```

## 关键点

1. **广播做布尔外积**。`pad_mask[:, :, None] & pad_mask[:, None, :]` 一步得到
   `(B, T, T)` 的相容矩阵，`out[b,i,j] = pad[b,i] & pad[b,j]`。这也是为什么 padding
   query 的整行会自动变 False——`q_keep` 那一项管着行。

2. **`tril` 默认含主对角线**。`diagonal=0` 表示保留 `j <= i`，即当前 token 能看到
   自己。若要「只能看严格更早的位置」，用 `diagonal=-1`；本题按常规因果语义包含
   对角线。

3. **约定 `True = 保留`，全局统一**。这和 SDPA 的 mask 约定一致（False 位置在
   softmax 前置 $-\infty$）。若反过来用 `True = 屏蔽`，所有布尔逻辑都要翻转，很容
   易出错，所以务必统一。

4. **传 `device` 避免设备不匹配**。`torch.ones(...)` 默认在 CPU；若 `pad_mask` 在
   GPU 上，直接做 `&` 会报 device mismatch。加 `device=pad_mask.device` 是好习惯。

5. **延伸**：这个 mask 就是喂给 `pytorch.llm.attention.scaled_dot_product_attention`
   和 `pytorch.llm.attention.mha` 的 `mask` 参数。padding + causal 的组合是 LLM 训
   练里高频的 bug 源，独立写对一次，后面直接复用。
