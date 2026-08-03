# 解题思路：Grouped-Query Attention (GQA)

## 一句话思路

分组查询注意力（grouped-query attention, GQA）是多头注意力（MHA）的省显存变体：
**多个 Q 头共享同一组 K/V 头**。实现上跟 MHA 几乎一样，只多一步——把数量较少的
K/V 头用 `repeat_interleave` 复制到和 Q 头一样多，后面的注意力就完全照旧。

## 拆解思路

### GQA 想省什么？

推理时要把历史 token 的 K/V 缓存下来（KV cache，见
`pytorch.llm.attention.kv_cache`），头越多缓存越大。GQA 让 K/V 只有
`num_kv_heads` 个头（少于 Q 的 `num_q_heads`），KV cache 就缩到原来的
`num_kv_heads / num_q_heads`。LLaMA-2 70B 用 64 个 Q 头、8 个 K/V 头，缓存直接
砍到 1/8。

```
MHA:  num_q_heads 个 Q 头, num_q_heads 个 K/V 头   （1:1）
GQA:  num_q_heads 个 Q 头, num_kv_heads 个 K/V 头  （num_kv_heads 整除 num_q_heads）
MQA:  num_q_heads 个 Q 头, 1 个 K/V 头             （多查询注意力，极端情形）
```

### 关键新操作：`repeat_interleave`

K/V 切头后是 `(B, num_kv_heads, T, head_dim)`，Q 是
`(B, num_q_heads, T, head_dim)`，头数对不上没法算注意力。用
`repeat_interleave(repeats, dim=1)` 把每个 K/V 头**原地重复** `repeats` 次：

```
原:      [kv0, kv1]
重复后:  [kv0, kv0, kv0, kv0, kv1, kv1, kv1, kv1]   （repeats=4）
```

注意和 `repeat` 不同——`repeat` 是 `[kv0, kv1, kv0, kv1, ...]`（整体循环）。这里
必须用 interleave，因为 Q 的头是**连续分组**的：Q 头 0~3 共享 K/V 头 0，Q 头
4~7 共享 K/V 头 1，以此类推。

## 参考实现

```python
import torch.nn.functional as F
from math import sqrt

def gqa(x, W_q, W_k, W_v, W_o, num_q_heads, num_kv_heads, mask=None):
    B, T, D = x.shape
    assert num_q_heads % num_kv_heads == 0
    repeats = num_q_heads // num_kv_heads
    head_dim = W_q.shape[1] // num_q_heads

    q, k, v = x @ W_q, x @ W_k, x @ W_v

    q = q.reshape(B, T, num_q_heads,  head_dim).transpose(1, 2)
    k = k.reshape(B, T, num_kv_heads, head_dim).transpose(1, 2)
    v = v.reshape(B, T, num_kv_heads, head_dim).transpose(1, 2)

    k = k.repeat_interleave(repeats, dim=1)     # 扩到 num_q_heads
    v = v.repeat_interleave(repeats, dim=1)

    scores = q @ k.transpose(-2, -1) / sqrt(head_dim)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    out = attn @ v

    out = out.transpose(1, 2).reshape(B, T, num_q_heads * head_dim)
    return out @ W_o
```

## 关键点

1. **用 `repeat_interleave` 而不是 `repeat`**。前者是 `[kv0, kv0, kv1, kv1]`（相
   邻复制），正好对应「Q 头连续分组共享一个 K/V 头」的语义；`repeat` 是整体循环
   `[kv0, kv1, kv0, kv1]`，会把 Q 头和 K/V 头的配对关系错开，数值就错了。

2. **`head_dim` 从 `W_q.shape[1]` 推导，不要写死 `D / num_q_heads`**。`W_q` 的输
   出维度是 `num_q_heads * head_dim`，直接除以 `num_q_heads` 最稳妥；有些模型的
   `head_dim` 并不等于 `D / num_q_heads`。

3. **`W_k` / `W_v` 比 `W_q` 窄**。它们的列数是 `num_kv_heads * head_dim`，这正是
   GQA 省参数、省 KV cache 的来源。

4. **退化情形自动兼容，不用写 `if`**。`num_kv_heads == num_q_heads` 时
   `repeats=1`，`repeat_interleave(1)` 是空操作，退化成标准 MHA；
   `num_kv_heads == 1` 时所有 Q 头共享一组 K/V，退化成 MQA。一份代码同时覆盖三种
   情况。

5. **mask 维度是 `num_q_heads`**。K/V 已经被扩到 `num_q_heads`，注意力打分是
   `(B, num_q_heads, T, T)`，mask 也按这个头数广播，别用 `num_kv_heads`。

6. **延伸**：GQA 只是把 `pytorch.llm.attention.mha` 的 K/V 头数调小再复制回来。
   它和 `pytorch.llm.attention.kv_cache` 是最大化推理省显存的黄金组合。
