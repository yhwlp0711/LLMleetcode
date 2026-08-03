# 解题思路：带 KV Cache 的 SDPA

## 一句话思路

自回归生成时，历史 token 的 K/V 是不变的。KV cache 就是把它们缓存起来，每步只算
**新 token 的 K/V**，拼到缓存末尾，再让新 query 对「完整 K/V」做一次注意力。核心
只有两步：**沿序列维拼接 cache + 标准 SDPA**，难点在处理首步 cache 为空的边界。

## 拆解思路

### 为什么需要 KV cache？

朴素写法每生成一个 token 都重算整个序列的注意力，$T$ 步累计是 $O(T^3)$，非常慢。
关键观察是：**已生成 token 的 K/V 只跟它自己有关，不会变**。于是每步只需要：

1. 算新 token 的 K/V（只有新增的几个位置，很小）；
2. 把新 K/V 拼到缓存的 K/V 末尾；
3. 用新 token 的 Q 对「完整缓存」做注意力。

单步从 $O(T^2)$ 降到 $O(T)$，长序列下差距巨大。

### 拼接沿哪一维？

张量形状是 `(B, H, T, D)`，要拼的是序列维 `T`，也就是 `dim=-2`。千万别拼 `dim=-1`
（那是特征维 `D`，拼错会让特征越变越长）。

### 首步没有历史

第一步 `k_cache` / `v_cache` 是 `None`（或空张量），此时直接用新的 K/V 当作完整
K/V，不做拼接。

## 参考实现

```python
import torch
import torch.nn.functional as F
from math import sqrt

def sdpa_with_kv_cache(q_new, k_new, v_new, k_cache, v_cache):
    if k_cache is not None and k_cache.numel() > 0:
        new_k_cache = torch.cat([k_cache, k_new], dim=-2)   # 沿 T 维拼接
        new_v_cache = torch.cat([v_cache, v_new], dim=-2)
    else:                                                    # 首步：cache 为空
        new_k_cache, new_v_cache = k_new, v_new

    d = q_new.shape[-1]
    scores = q_new @ new_k_cache.transpose(-2, -1) / sqrt(d)  # q 对完整 K
    attn = F.softmax(scores, dim=-1)
    out = attn @ new_v_cache

    return out, new_k_cache, new_v_cache
```

## 关键点

1. **拼接用 `dim=-2`（序列维 T）**。形状 `(B, H, T, D)` 里 `T` 在倒数第二维。用相
   对索引 `-2` 比写死 `2` 更稳，换布局也不出错。

2. **正确处理 `cache=None` 的边界**。首步没有历史，`if k_cache is not None and
   k_cache.numel() > 0` 同时兼容「传 None」和「传空张量」两种约定；否则用新 K/V
   直接当完整 K/V。

3. **SDPA 用「拼接后的完整 K/V」**。`q_new` 可能只有 1 个 token，但它要看到全部历
   史，所以 key/value 用拼好的 `new_k_cache` / `new_v_cache`。形状对账：
   `(B,H,T_new,D) @ (B,H,D,T_full) → (B,H,T_new,T_full)`。KV cache 的本质就是 q
   短、k/v 越来越长。

4. **正确性的等价定义：增量 == 一次性 prefill**。把序列一次性喂进去算出的最后一
   步输出，应当等于「先填好前 $T-1$ 步的 cache，再增量算第 $T$ 步」的输出。因为
   最后一步的 query 是同一个、能看到的 K/V 也是同一批，两者数学上完全等价。对不
   上就说明 cache 拼错了。本题按题面简化**不加 mask**（构造场景保证 q 只看 ≤ 自
   己位置的 key）。

5. **延伸**：这里用 `torch.cat` 每步重新分配，简单但有额外开销；工业实现常预分配
   buffer 再写入。KV cache 和 `pytorch.llm.attention.gqa` 一起用能最大化省显存，
   底层的裸注意力见 `pytorch.llm.attention.scaled_dot_product_attention`。
