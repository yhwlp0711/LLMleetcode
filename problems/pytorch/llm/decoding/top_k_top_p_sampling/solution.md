# 解题思路：Top-k / Top-p Sampling

## 一句话思路

这题按顺序做三道过滤：**温度缩放（temperature）→ top-k → top-p（nucleus，核采
样）**，把「不该被采样到」的 token 的 logit 置为 $-\infty$，返回过滤后的 logits
（真正的随机采样留给调用方）。难点在 top-p 的「排序 + 累积概率 + 右移一位 + 散
回原索引」。

## 拆解思路

三道过滤各管一件事：

| 过滤 | 做什么 | 作用 |
|---|---|---|
| Temperature | `logits / T` | 调随机程度：>1 更平（更随机），<1 更尖（更确定）|
| Top-k | 只留最大的 k 个 | 硬性限制候选数量 |
| Top-p | 留累积概率刚够 p 的最小集合 | 按分布形状动态限制 |

### Top-k：找第 k 大当阈值

`topk(k)` 返回排好序的前 k 个值，取其中最后一个（`[..., -1]`）就是「第 k 大」的
阈值。凡是小于这个阈值的 logit 全置 $-\infty$。加 `None` 是为了让阈值形状变成
`(B, 1)`，方便和 `(B, V)` 的 logits 广播比较。

### Top-p：核心是「右移一位」

先把 logits 降序排序、softmax 成概率、算累积概率（cumsum）。然后要移除「累积概率
**超过** `top_p` 之后」的 token。看个例子，概率 `[0.4, 0.3, 0.2, 0.1]`，
`top_p=0.5`：

- 累积：`[0.4, 0.7, 0.9, 1.0]`
- 直接标记 `> 0.5`：`[F, T, T, T]`——但这样只保留了第 1 个（0.4），而核采样要求
  **保留刚刚让累积超过 p 的那个 token**（第 2 个）。

所以把移除标记**右移一位**：`[F, T, T, T]` → `[F, F, T, T]`，并强制第 0 位为
`False`（至少保留 1 个 token，防止最大概率就 > p 时全被屏蔽）。

### 散回原索引

上面的移除标记是「排序后」的顺序，但要作用到原始顺序的 logits 上。用
`scatter_` 按 `sorted_idx` 把标记写回原位置：`remove_mask[sorted_idx[i]] =
sorted_remove[i]`。这是 `sort` 之后的标准还原技巧。

## 参考实现

```python
import torch

def filter_logits(logits, *, temperature=1.0, top_k=0, top_p=1.0):
    out = logits / temperature                              # 1. 温度缩放

    if top_k > 0:                                           # 2. top-k
        k = min(top_k, out.shape[-1])
        kth = out.topk(k, dim=-1).values[..., -1, None]     # 每行第 k 大值
        out = torch.where(out < kth, torch.full_like(out, float("-inf")), out)

    if top_p < 1.0:                                         # 3. top-p
        sorted_logits, sorted_idx = out.sort(dim=-1, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum_probs = probs.cumsum(dim=-1)

        sorted_remove = cum_probs > top_p                   # 超过 p 的
        sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()  # 右移一位
        sorted_remove[..., 0] = False                       # 至少保留 1 个

        remove_mask = torch.zeros_like(sorted_remove)
        remove_mask.scatter_(dim=-1, index=sorted_idx, src=sorted_remove)  # 散回原索引
        out = out.masked_fill(remove_mask, float("-inf"))

    return out
```

## 关键点

1. **过滤顺序固定：temperature → top-k → top-p**。温度先改变整体尖锐度，再依次做
   两道候选筛选。

2. **top-k 用「第 k 大值」当阈值**。`topk(k).values[..., -1, None]` 取到每行第 k
   大值并加一维，再用 `torch.where(out < kth, -inf, out)` 屏蔽小于它的全部 logit。

3. **top-p 的「右移一位」是灵魂**。要保留「刚让累积概率超过 p 的那个 token」，所以
   把 `cum_probs > top_p` 的标记整体右移，并把第 0 位设 `False` 保证至少留 1 个。

4. **右移赋值必须 `.clone()`**。`sorted_remove[..., 1:] = sorted_remove[..., :-1]`
   源和目标内存重叠（aliasing），不 clone 会边写边读产生错误结果。

5. **`scatter_` 把排序后的标记散回原索引**。排序打乱了顺序，`scatter_(dim=-1,
   index=sorted_idx, src=...)` 按原始位置还原移除标记，最后 `masked_fill` 屏蔽。

6. **延伸**：只返回过滤后的 logits、不做实际采样，是因为 `torch.multinomial` 涉及
   随机数、难以精确判分。调用方拿到过滤结果后自行 `softmax` + `multinomial` 采样。
   固定选最大值不带随机的版本就是 `pytorch.llm.decoding.greedy_decode`。
