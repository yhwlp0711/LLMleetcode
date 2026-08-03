# 解题思路：Scaled Dot-Product Attention

## 一句话思路

缩放点积注意力（scaled dot-product attention, SDPA）就是让每个 query 去和所有
key 算「相似度」，softmax 成权重后，对 value 做加权平均。核心公式一行，关键在
**除以 $\sqrt{D}$ 的缩放**、**mask 的加性应用**和 **softmax 的数值稳定
（numerical stability）**。

## 从直觉到公式

### 注意力在做什么？

把每个 query 想成「我想找什么信息」，每个 key 是「我这里有什么」，value 是「我
实际能给出的内容」。做法很自然：

1. 用 query 和每个 key 做点积，点积越大说明越「匹配」，得到打分 $QK^\top$。
2. 把打分过一遍 softmax，变成一组和为 1 的权重。
3. 用这组权重去加权平均 value，就是这个 query「读到」的信息。

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{D}} + \text{Mask}\right) V$$

### 为什么要除以 $\sqrt{D}$？

点积是 $D$ 个数相乘再相加。$D$ 越大，点积的数值波动（方差）就越大，softmax 容
易被推到「几乎全押在一个位置」的极端状态，梯度变得很小、难训练。除以 $\sqrt{D}$
正好把方差拉回到 1 附近，让 softmax 保持在「有区分但不极端」的区间。

### mask 怎么用？

mask 用来屏蔽「不该看的位置」（比如未来的 token、padding）。约定 `True = 保留`。
做法是把 `False` 的位置在 softmax **之前**加上 $-\infty$——这样 `exp(-inf)=0`，
softmax 后这些位置权重正好是 0。

## 参考实现

```python
import torch.nn.functional as F
from math import sqrt

def sdpa(q, k, v, mask=None):
    d = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / sqrt(d)          # (B, H, T_q, T_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))  # False 位置置 -inf
    attn = F.softmax(scores, dim=-1)                    # 沿 key 维归一化
    return attn @ v                                     # 加权平均 value
```

## 关键点

1. **缩放放在 softmax 之前**。$\sqrt{D}$ 的作用是稳定 softmax 的输入尺度，所以
   必须先除再 softmax，顺序不能反。注意是 key 维度 $D$（`q.shape[-1]`），不是序
   列长度。

2. **`k.transpose(-2, -1)` 只转置最后两维**。不能用 `.T`——对 4D 张量 `.T` 会把
   全部维度反转，前面的 batch、head 维必须保持不动，才能对每个 batch、每个头独
   立算注意力。形状对账：`(B,H,T_q,D) @ (B,H,D,T_k) → (B,H,T_q,T_k)`。

3. **mask 是加性的，且在 softmax 之前**。约定 `True = 保留`，所以 `masked_fill`
   要对 `~mask`（False 位置）填 $-\infty$，softmax 后它们自然变 0；如果放到
   softmax 之后再置 0，剩下的权重就不再和为 1 了。用 $-\infty$ 而不是「很大的负
   数」（如 `-1e9`），因为半精度下 `-1e9` 不够小、softmax 后仍有残留。

4. **softmax 沿最后一维（key 维）归一化**。每个 query 对所有 key 的权重加起来才
   等于 1。`F.softmax` 内部已做「减去每行 max」的稳定处理（见
   `pytorch.nn.numeric_activations`），不用自己减 max。

5. **延伸**：SDPA 是所有注意力变体的基石。多头版本见 `pytorch.llm.attention.mha`，
   多个 query 头共享 K/V 的省显存版本见 `pytorch.llm.attention.gqa`，推理时缓存
   历史 K/V 的加速版本见 `pytorch.llm.attention.kv_cache`。
