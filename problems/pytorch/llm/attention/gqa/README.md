# Grouped-Query Attention (GQA)

实现 GQA —— **多个 Q head 共享同一组 K/V head** 的注意力变体。LLaMA-2/3、
Mistral、Qwen 都用。

## 函数签名

```python
def gqa(
    x: torch.Tensor,             # (B, T, D)
    W_q: torch.Tensor,           # (D, num_q_heads * head_dim)
    W_k: torch.Tensor,           # (D, num_kv_heads * head_dim)
    W_v: torch.Tensor,           # (D, num_kv_heads * head_dim)
    W_o: torch.Tensor,           # (num_q_heads * head_dim, D)
    num_q_heads: int,
    num_kv_heads: int,           # 须能整除 num_q_heads
    mask: torch.Tensor | None = None,  # (B, num_q_heads, T, T)
) -> torch.Tensor:               # (B, T, D)
```

## 算法（在 MHA 基础上）

1. **投影**：q = x @ W_q（`num_q_heads * head_dim` 输出）；k/v 用更小的
   `num_kv_heads * head_dim`。
2. **切头**：
   - q reshape 到 `(B, T, num_q_heads, head_dim)` → transpose 到
     `(B, num_q_heads, T, head_dim)`
   - k, v reshape 到 `(B, T, num_kv_heads, head_dim)` → transpose 到
     `(B, num_kv_heads, T, head_dim)`
3. **重复 K/V**：用 `repeat_interleave` 把 K/V 头数从 `num_kv_heads` 扩
   到 `num_q_heads`：每个 K/V head 重复 `num_q_heads // num_kv_heads` 次。
4. **SDPA**：跟普通 MHA 一样。
5. **合头 + 输出投影**：跟 MHA 一样。

## 退化情形

- `num_kv_heads == num_q_heads`：退化为标准 MHA。
- `num_kv_heads == 1`：退化为 MQA（Multi-Query Attention）。

判分会同时测试这两个退化点。

## 说明

- mask 约定：`True = 保留`。
- W 矩阵都不带 bias。
- 容差 `atol=1e-5`。
