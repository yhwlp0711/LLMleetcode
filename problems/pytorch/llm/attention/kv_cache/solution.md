# 解题思路：带 KV Cache 的 SDPA

## 为什么需要 KV Cache

LLM 自回归生成时，每生成一个新 token，需要重新计算「整个已生成序列」的
注意力。Naive 写法每步 O(T²)，T 步总共 O(T³)，巨慢。

观察：**已生成 token 的 K/V 是不变的**。把它们缓存下来，每步只需：
1. 算新 token 的 K/V（很小，只有 1 个位置）
2. 把新 K/V append 到 cache
3. 新 token 的 Q 跟「整个 cache」做 attention（O(T) 而非 O(T²)）

总复杂度从 O(T³) 降到 O(T²)，对长序列差异巨大。

## 参考实现

```python
def sdpa_with_kv_cache(q_new, k_new, v_new, k_cache, v_cache):
    if k_cache is not None and k_cache.numel() > 0:
        new_k = torch.cat([k_cache, k_new], dim=-2)
        new_v = torch.cat([v_cache, v_new], dim=-2)
    else:
        new_k = k_new
        new_v = v_new

    d = q_new.shape[-1]
    scores = q_new @ new_k.transpose(-2, -1) / sqrt(d)
    attn = F.softmax(scores, dim=-1)
    out = attn @ new_v

    return out, new_k, new_v
```

## 关键点

### 1. `dim=-2` 拼接的是 T 维

张量 shape `(B, H, T, D)`，`dim=-2 = T 维`。**不要用 `dim=-1`**（那是 D
维，拼起来会让特征维变长，完全错）。

### 2. 处理 `cache=None` 的边界

首步调用时没有历史，cache 是 `None`（或长度为 0 的张量）。简单 `if` 判
断即可：

```python
if k_cache is not None and k_cache.numel() > 0:
    new_k = torch.cat([k_cache, k_new], dim=-2)
else:
    new_k = k_new
```

加 `numel() > 0` 是为了同时兼容「传 None」和「传空张量」两种约定。

### 3. SDPA 用「拼接后的完整 K/V」

`q_new` 是「本步的 query」（很可能只有 1 个 token），但它要看**整个**
历史 + 当前 → key 用 `new_k`（包含 cache）。这是 KV cache 的本质：q 不
变，k/v 越来越长。

形状对账：`(B, H, T_new, D) @ (B, H, D, T_full) → (B, H, T_new, T_full)`。

## 「增量 == prefill」属性测试详解

这是判分的灵魂。原理：

```
prefill 方式：
    out = sdpa(q[0:T], k[0:T], v[0:T])    # 一次性算 T 步
    out[-1] 是最后一步的输出

incremental 方式：
    填好 cache，cache = (k[0:T-1], v[0:T-1])
    out_last = sdpa_with_kv_cache(q[T-1:T], k[T-1:T], v[T-1:T], k_cache, v_cache)
```

两者数学上**完全等价**，因为：
- 最后一步的 query = `q[T-1]`
- 它能看到的所有 key/value = `k[0:T]` / `v[0:T]`
- attention 算的就是这一组

不等价就说明 KV cache 实现错了。这是面试官最爱问的「自洽性」测试。

## 易错点

### 1. `dim=-2` vs `dim=2`

如果你硬编码 `dim=2`，对 `(B, H, T, D)` 也对（T 在 axis 2）。但有人写
GQA / multi-query 时可能改 shape 布局，用相对索引 `-2` 更鲁棒。

### 2. cache 是否要分离 K 和 V

本题用两个 tensor 分开传。也有实现把 K/V 合成一个 5D tensor `(2, B, H, T, D)`
或一个 dataclass `KVCache(k, v)`。判分会跟参考实现对齐，所以照题面来。

### 3. cache 是否要更新内存

题目让你**返回** new_k_cache（不要求原地 append）。实际工业代码可能用预
分配的 buffer + index_copy_ 写入，更省内存。本题简化为 `torch.cat`，每
步重新分配。这对正确性没影响，性能差。

## 跟 Flash Attention 的关系

KV cache 是「inference 时」的优化；Flash Attention 是「training 时」的
内存优化（分块算 softmax 不暂存大 attention matrix）。两者正交，可以同
时用 —— 比如 vLLM 就是 KV cache + 类似 flash 的算子。
