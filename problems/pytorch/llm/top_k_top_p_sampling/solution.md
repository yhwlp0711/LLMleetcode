# 解题思路：Top-k / Top-p Sampling

## 三个过滤的语义

| 过滤 | 作用 | 控制目的 |
|---|---|---|
| **Temperature** | 缩放整个 logits 分布 | 调节随机程度 |
| **Top-k** | 只保留前 k 大的 logit | 限制候选数量（hard cutoff）|
| **Top-p** (nucleus) | 保留累积概率达到 p 的最小集合 | 动态限制（基于分布形状）|

实际推理一般组合使用：先 temperature → top-k 粗筛 → top-p 精筛 → softmax
→ multinomial 采样。本题只做前 3 步，把采样留给调用方。

## 参考实现

```python
def filter_logits(logits, *, temperature=1.0, top_k=0, top_p=1.0):
    out = logits / temperature

    if top_k > 0:
        k = min(top_k, out.shape[-1])
        kth = out.topk(k, dim=-1).values[..., -1, None]   # 每行第 k 大值
        out = torch.where(out < kth, torch.full_like(out, float("-inf")), out)

    if top_p < 1.0:
        sorted_logits, sorted_idx = out.sort(dim=-1, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum_probs = probs.cumsum(dim=-1)

        sorted_remove = cum_probs > top_p
        sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()  # 右移
        sorted_remove[..., 0] = False                              # 保 1

        remove_mask = torch.zeros_like(sorted_remove)
        remove_mask.scatter_(dim=-1, index=sorted_idx, src=sorted_remove)
        out = out.masked_fill(remove_mask, float("-inf"))

    return out
```

## Top-k 的关键技巧

`topk` 返回排好序的前 k 个，取最后一个就是「第 k 大的值」—— 这是阈值。
小于阈值的全置 `-inf`。

```python
kth = out.topk(k, dim=-1).values[..., -1, None]  # (..., 1) 用于广播
out = torch.where(out < kth, -inf, out)
```

为什么要 `[..., -1, None]`？`-1` 取第 k 个（即最小的入选值），`None` 加
一个维度方便和 `out` 广播。

## Top-p 的核心难点：**「右移一位」**

考虑一个例子：sorted 概率 `[0.4, 0.3, 0.2, 0.1]`，`top_p = 0.5`：
- 累积概率 `[0.4, 0.7, 0.9, 1.0]`
- 哪些 `> 0.5`? 索引 `[F, T, T, T]`

直接屏蔽这些会**只保留第 1 个**（0.4 < 0.5）—— 但根据 Nucleus Sampling
论文，应该**保留刚刚超过 p 的 token**，即第 2 个（累积 0.7 > 0.5 时也要
包含）。

所以要把 mask **右移一位**：
- 右移前：`[F, T, T, T]`
- 右移后：`[F, F, T, T]`

第 1 位强制 `False`（必保留至少 1 个），保证哪怕第 1 个的概率已经 > p
也不会全屏蔽。

```python
sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
sorted_remove[..., 0] = False
```

**`.clone()` 非常关键**：直接 `sorted_remove[..., 1:] = sorted_remove[..., :-1]`
会因为内存重叠产生 aliasing bug。必须先 clone 再赋值。

## 把排序后的 mask 散回原索引

经过 `sort + cumsum + 右移`，我们得到的 `sorted_remove` 是**按排序后位置
索引的**。但 `out` 还是原始顺序。要把这个 mask 还原到原索引：

```python
remove_mask = torch.zeros_like(sorted_remove)
remove_mask.scatter_(dim=-1, index=sorted_idx, src=sorted_remove)
```

`scatter_` 的语义：`remove_mask[sorted_idx[i]] = sorted_remove[i]`。也就是
把「排序后位置 i 是否要移除」的信息，按 `sorted_idx[i]`（原索引）写回。

这是 `sort` 之后**必学的还原技巧**。

## 为什么不直接做采样？

`torch.multinomial(probs, num_samples=1)` 涉及随机数，跟用户实现的 RNG
状态强相关 → 难以判分（同 seed 但 RNG 消耗顺序不同就会不一样）。

把"过滤"独立成纯函数，**确定性可判分**，把"采样"留给调用方：

```python
# 用户在自己代码里组合
filtered = filter_logits(logits, top_k=50, top_p=0.95)
probs = filtered.softmax(dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
```

这种「纯函数 + 用户拼装」是判分友好的设计哲学。

## 性能小贴士

`out.sort(dim=-1)` 是 O(V log V)；如果只关心 nucleus 阈值，理论上可以用
`torch.topk + 累积`，但实际差距不大，可读性优先。
