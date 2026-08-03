# 带 KV Cache 的 SDPA

实现支持 KV cache 的注意力 forward —— LLM autoregressive 推理的核心优化。

## 函数签名

```python
def sdpa_with_kv_cache(
    q_new: torch.Tensor,                     # (B, H, T_new, D)  本步新的 query
    k_new: torch.Tensor,                     # (B, H, T_new, D)  本步新的 key
    v_new: torch.Tensor,                     # (B, H, T_new, D)  本步新的 value
    k_cache: torch.Tensor | None,            # (B, H, T_past, D) 历史 key（首步可为 None 或空）
    v_cache: torch.Tensor | None,            # (B, H, T_past, D)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    返回:
        out:         (B, H, T_new, D)  attention 输出
        new_k_cache: (B, H, T_past + T_new, D)  更新后的 key cache
        new_v_cache: (B, H, T_past + T_new, D)  更新后的 value cache
    """
```

## 算法

1. **拼接 cache**：把 `k_new` / `v_new` 沿 `T` 维拼到 `k_cache` / `v_cache`
   末尾，得到新的完整 K/V。
2. **SDPA**：用 `q_new` 与拼接后的完整 K/V 算注意力（**无 mask** 或自动
   causal，本题简化为「不带 mask」—— 题目保证 K 序列里没有未来 token）。
3. **返回**：attention 输出 + 更新后的 K cache + 更新后的 V cache。

## 关键性质（也是判分核心）

**「逐 token 增量 == 一次性 prefill」**：

```
方式 A (prefill):
    out_A = sdpa(q_full, k_full, v_full)

方式 B (incremental):
    out, k, v = sdpa_with_kv_cache(q_full[:1], k_full[:1], v_full[:1], None, None)
    for i in range(1, T):
        out, k, v = sdpa_with_kv_cache(q_full[i:i+1], k_full[i:i+1], v_full[i:i+1], k, v)
    out_B = 最后一次的 out
```

应当满足 `out_B == out_A[-1:]`。这是 KV cache 实现正确性的等价定义。

## 说明

- `k_cache` / `v_cache` 为 `None` 表示首步（空 cache）；用户要处理这种边
  界情况。
- 简化起见**不要 mask**（判分构造场景保证 q 只看 ≤ 自己位置的 key）。
- SDPA 的 scaling 是 `sqrt(D)`，使用 stable softmax。
- 容差 `atol=1e-5`。
