# 解题思路：Scaled Dot-Product Attention

## 参考实现

```python
from math import sqrt
import torch.nn.functional as F

def sdpa(q, k, v, mask=None):
    d = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / sqrt(d)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    return attn @ v
```

整个公式 4 个步骤，每一步都对应一行代码。

## 步骤拆解

### 步骤 1：相似度分数 $QK^\top$

`q` shape `(B, H, T_q, D)`，`k` shape `(B, H, T_k, D)`。要求点积成 shape
`(B, H, T_q, T_k)`，即 `q @ k.transpose(-2, -1)`。

**注意**：`transpose(-2, -1)` 而不是 `.T`。`.T` 是全维度反转，对 4D 张量
会变成 `(D, T_k, H, B)`，完全错误。`transpose(-2, -1)` 只交换最后两维。

### 步骤 2：缩放 $\sqrt{D}$

为什么要除 $\sqrt{D}$？原论文的解释：当 $D$ 大时，$QK^\top$ 的方差正比于
$D$，softmax 容易进入饱和区（梯度消失）。除以 $\sqrt{D}$ 把方差稳定在 1
附近。

**注意**：是 $\sqrt{D}$ 不是 $\sqrt{T}$；用 key 维度，不是序列长度。

### 步骤 3：mask

`scores.masked_fill(~mask, -inf)`：把 `mask` 为 `False`（取反后 `True`）
的位置填 $-\infty$。softmax 中 $e^{-\infty} = 0$，所以这些位置不贡献。

**关键技巧**：用 `-inf` 而不是「很大的负数」（如 `-1e9`）。在 fp16 / bf16
下 `-1e9` 不够小，softmax 后仍有微小残留；`-inf` 在 fp16 下也是合法值，安
全可靠。

**易错点**：`mask` 是 bool 类型，`~mask` 是位取反。如果题目用「True = 屏
蔽」（反向约定），就直接 `masked_fill(mask, -inf)`，不要 `~`。本题 `True
= 保留`，要 `~`。

### 步骤 4：softmax + 加权求和

`F.softmax(scores, dim=-1)` —— 沿 key 维归一化成概率分布。PyTorch 的实现
已经是数值稳定的（内部减最大值），不需要自己减。

最后 `attn @ v`：每个 query 位置对所有 value 做加权平均。形状对账：
`(B, H, T_q, T_k) @ (B, H, T_k, D_v) → (B, H, T_q, D_v)`。

## 完整流程图

```
  q (B,H,T_q,D)    k (B,H,T_k,D)         v (B,H,T_k,D_v)
        \              |                       |
         \    transpose(-2,-1)                |
          \            |                       |
           +→ @ → scores (B,H,T_q,T_k)        |
                       |                       |
              / sqrt(D)                       |
                       |                       |
                  + mask (-inf)               |
                       |                       |
                  softmax(-1)                 |
                       |                       |
                       +→ @ → out (B,H,T_q,D_v)
```

## 为什么单独考这个？

SDPA 是 Transformer 一切注意力变体的内核：
- Multi-Head Attention 在它之上加投影 + reshape
- Grouped-Query Attention 在它之上让多个 query head 共享 k/v
- Flash Attention 是 SDPA 的 IO-aware 高效实现

理解了 SDPA，所有注意力变体都是套壳。
