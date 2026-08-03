# 因果 + Padding Mask

构造 Transformer attention 用的布尔 mask，**同时**处理因果约束和 padding。
看似简单，实际上 padding mask + causal mask 的组合是 LLM 推理代码里最常出
bug 的地方之一。

## 函数签名

```python
def build_attention_mask(
    pad_mask: torch.Tensor,   # (B, T) bool；True = 真实 token，False = padding
    causal: bool,             # 是否叠加 causal 约束
) -> torch.Tensor:            # (B, 1, T, T) bool；True = 保留，False = 屏蔽
```

## 语义

输出 `out[b, 0, i, j] = True` 当且仅当**所有**条件成立：

1. `pad_mask[b, i] == True`（query 位置 i 不是 padding）
2. `pad_mask[b, j] == True`（key   位置 j 不是 padding）
3. 如果 `causal=True`，还要求 `j <= i`

例：`pad_mask = [True, True, True, False]`，`causal=True`：

```
output[0, 0]:
    j=0  j=1  j=2  j=3
i=0  T    F    F    F
i=1  T    T    F    F
i=2  T    T    T    F
i=3  F    F    F    F
```

第 i=3 行整行 False，因为 query 是 padding；第 j=3 列除了 i=3 也全是 F。

## 说明

- 输入 `pad_mask` 是 `torch.bool`，shape `(B, T)`。
- 输出必须是 `torch.bool`，shape `(B, 1, T, T)` —— 中间的 `1` 是 head 维
  占位，跟 `sdpa` 的 mask 约定一致。
- **不要用 Python 循环**，全程用 broadcasting + 布尔运算。
