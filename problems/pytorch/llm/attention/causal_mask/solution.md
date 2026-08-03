# 解题思路：因果 + Padding Mask

## 一句话理解

把两个独立的「保留集」做交集：

- **Padding 保留集**：query 和 key 都不是 padding
- **Causal 保留集**：key 的位置不超过 query 的位置

## 参考实现

```python
def build_attention_mask(pad_mask, causal):
    B, T = pad_mask.shape
    q_keep = pad_mask[:, :, None]   # (B, T, 1) — i 维
    k_keep = pad_mask[:, None, :]   # (B, 1, T) — j 维
    base = q_keep & k_keep          # (B, T, T) — 两端都非 padding

    if causal:
        tri = torch.tril(torch.ones(T, T, dtype=torch.bool, device=pad_mask.device))
        base = base & tri

    return base.unsqueeze(1)        # (B, 1, T, T)
```

## 关键技巧

### 1. 用广播构造 (T, T) 的相容矩阵

```python
q_keep = pad_mask[:, :, None]   # (B, T) → (B, T, 1)
k_keep = pad_mask[:, None, :]   # (B, T) → (B, 1, T)
q_keep & k_keep                 # (B, T, 1) & (B, 1, T) → (B, T, T)
```

这是经典「外积式」操作的布尔版本。`out[b, i, j] = pad[b, i] & pad[b, j]`，
正好对应「(i, j) 这格保留 ⟺ 两端都是真实 token」。

### 2. `torch.tril(torch.ones(T, T, bool))` 生成 causal mask

下三角矩阵：

```
[[T, F, F, F],
 [T, T, F, F],
 [T, T, T, F],
 [T, T, T, T]]
```

`tril` 默认 `diagonal=0`，即**包括主对角线**。如果想"严格小于 i"（不能看
当前位置），用 `diagonal=-1`。本题按常规 causal 语义，包含对角线。

### 3. unsqueeze(1) 加 head 维

输出要求 `(B, 1, T, T)`，其中 `1` 是 head 维的占位 —— 后续 SDPA 会广播到
`(B, num_heads, T, T)`。如果不加这个 head 维，SDPA 里 `scores.masked_fill(~mask, -inf)`
就 shape 对不上了。

## 易错点

### 1. `pad_mask` 是 bool 而不是 float

按惯例 `True = 保留`，跟 SDPA 题约定一致。如果你想反过来（`True = 屏蔽`），
所有逻辑要翻转，容易出错。**统一一个约定**，全局严守。

### 2. causal mask 要不要 query 维度也来 mask？

经典写法是 `j <= i`，跟 `i` 的真实性没关系。但题目要求**整行**也清空（i
是 padding 的行），所以最终结果里 padding query 的整行都是 F。这是
`q_keep[:, :, None]` 这一项的贡献。

### 3. device 要传

`torch.ones(T, T)` 默认在 CPU 上，如果 `pad_mask` 在 GPU 上，做 `&` 会报
device mismatch。所以参考实现里加了 `device=pad_mask.device`。本题判分都
在 CPU，但养成习惯总没错。

## 为什么这道题值得单独练？

`pad + causal mask` 的组合是 LLM 训练代码里的高频 bug 源：

- 忘记把 padding query 的整行清空 → loss 计算时把无效位置算进去
- 反向约定 `True = 屏蔽` 跟下游 SDPA 不一致
- mask 没加 head 维度 → 广播失败

把它独立成题，逼你写对一次，未来直接复用。
