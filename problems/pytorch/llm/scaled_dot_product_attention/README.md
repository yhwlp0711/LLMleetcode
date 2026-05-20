# Scaled Dot-Product Attention

实现 **scaled dot-product attention** —— 每个 Transformer 的核心计算。

## 函数签名

```python
def sdpa(
    q: torch.Tensor,                  # (B, H, T_q, D)
    k: torch.Tensor,                  # (B, H, T_k, D)
    v: torch.Tensor,                  # (B, H, T_k, D_v)
    mask: torch.Tensor | None = None, # 可广播到 (B, H, T_q, T_k)；True = 保留
) -> torch.Tensor:                    # (B, H, T_q, D_v)
```

## 公式

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{D}} + \text{Mask}\right) V$$

其中：
- $D$ 是 key 维度（`q.shape[-1]`）。
- mask **加性应用** —— `mask` 为 `False`（或 `0`）的位置在 softmax 前置为
  $-\infty$。`mask=None` 跳过 mask。
- softmax 必须**数值稳定**（先减去每行 max 再 exp）。

## 说明

- 判分用 shape `(B, 1, T_q, T_k)` 的布尔 mask（典型的 causal / padding mask
  布局）。
- **不做 dropout**（这是裸算子）。
- 想用什么 PyTorch op 都行（`F.softmax`、`torch.matmul` 等），但**不要直接
  调用** `F.scaled_dot_product_attention` —— 那就违背了练手的本意。

## 提示

正确实现大约 5 行。
