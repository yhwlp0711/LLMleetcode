# Multi-Head Attention（纯函数版）

实现多头注意力，**作为纯函数** —— 所有权重作为参数传入，不用 `nn.Module`。
这样把「注意力算法本身」（投影 → 切分多头 → SDPA → 合并多头 → 输出投影）
和「类/init」概念解耦。

## 函数签名

```python
def mha(
    x: torch.Tensor,                 # (B, T, D)   输入序列
    W_q: torch.Tensor,               # (D, D)      query 投影权重
    W_k: torch.Tensor,               # (D, D)      key 投影权重
    W_v: torch.Tensor,               # (D, D)      value 投影权重
    W_o: torch.Tensor,               # (D, D)      输出投影权重
    num_heads: int,                  # 头数（D 必须能整除）
    mask: torch.Tensor | None = None, # 可广播到 (B, num_heads, T, T)；True = 保留
) -> torch.Tensor:                   # (B, T, D)   注意力输出
```

## 算法

1. **Q/K/V 投影**：`q = x @ W_q`、`k = x @ W_k`、`v = x @ W_v`，shape 都是
   `(B, T, D)`。
2. **切分多头**：reshape 到 `(B, T, num_heads, head_dim)`，再 transpose 到
   `(B, num_heads, T, head_dim)`。其中 `head_dim = D / num_heads`。
3. **每头做 SDPA**：
   $\text{softmax}\big(QK^\top / \sqrt{\text{head\\_dim}} + \text{mask}\big) V$
4. **合并多头**：transpose 回 `(B, T, num_heads, head_dim)`，再 reshape 成
   `(B, T, D)`。
5. **输出投影**：`out = merged @ W_o`，shape `(B, T, D)`。

## 说明

- mask 约定与 `pytorch.llm.scaled_dot_product_attention` 一致：`True = 保
  留`，`False = 屏蔽`（softmax 前置为 `-inf`）。
- **不带 bias**（本题省略 Q/K/V/O 的 bias 项）。
- 判分用固定 seed 构造权重张量，所以直接 `atol=1e-5` 数值对比。
- 不要调用 `nn.MultiheadAttention` 或 `F.scaled_dot_product_attention`；只
  用 `matmul`、`softmax`、`reshape`/`view`、`transpose`。
