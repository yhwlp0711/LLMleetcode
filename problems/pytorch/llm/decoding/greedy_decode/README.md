# Greedy Decode

实现最经典的 autoregressive 解码算法 —— 贪心解码。**「模型」抽象成一个函数** ——
给定 input_ids 返回下一步 logits，无需真正加载 LM。

## 待实现函数

```python
def greedy_decode(
    model_fn: Callable[[torch.Tensor], torch.Tensor],   # ids (1, T) -> logits (1, V)
    input_ids: torch.Tensor,    # (1, T_init)
    max_len: int,               # 最大生成长度（不含 prompt）
    eos_id: int,
) -> torch.Tensor:              # (1, T_init + n_generated)
```

每步取 `argmax(logits)`，append 到序列；遇到 `eos_id` 立即停止；最长生成
`max_len` 个新 token。

## `model_fn` 的语义

判分会传入一个**确定性**的 `model_fn`，行为类似查表：

```python
# 同样的 input_ids 调用，永远返回同样的 logits
logits = model_fn(input_ids)   # input_ids: (B, T), logits: (B, V)
```

只看最后一个 token 的位置预测下一个 token —— `model_fn` 已经处理好这部分。

## 说明

- 输入 `input_ids` 永远是 `(1, T_init)`（单 prompt 起始）。
- 词表大小 ≤ 50，max_len ≤ 16 —— 控制运行时间。
- `eos_id` 是约定的 EOS token。如果一直没出现 eos，跑满 max_len 后返回。
- 进阶版（Beam Search）见 `pytorch.llm.decoding.beam_search`。
