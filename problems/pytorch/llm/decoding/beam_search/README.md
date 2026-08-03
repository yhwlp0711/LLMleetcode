# Beam Search

实现 beam search 解码。**「模型」抽象成一个函数** —— 给定 input_ids 返回下一步
logits，无需真正加载 LM。

> 更简单的贪心解码见 `pytorch.llm.decoding.greedy_decode`（beam_size=1 的特例）。

## 待实现函数

```python
def beam_search(
    model_fn: Callable[[torch.Tensor], torch.Tensor],   # ids (B, T) -> logits (B, V)
    input_ids: torch.Tensor,    # (1, T_init)
    max_len: int,
    beam_size: int,
    eos_id: int,
) -> torch.Tensor:              # (1, T_init + n_generated)  最优 beam
```

经典 beam search：

1. 第一步用 prompt 算 logits，取 top-`beam_size` 作为初始 beams
2. 每步扩展所有 beams 到所有候选 token：得到 `beam_size × vocab` 个候选
3. 取累积 log-prob 前 `beam_size` 个，剪枝
4. 跟踪每个 beam 是否已结束（遇到 eos）；已结束的 beam 不再扩展
5. 最长 `max_len` 步；返回**最终累积 log-prob 最大**的 beam

**长度归一化**：本题用 `score = sum_log_probs / length` 来比较已结束的
beam（length = 生成长度，不含 prompt）。

## `model_fn` 的语义

判分会传入一个**确定性**的 `model_fn`，行为类似查表：

```python
# 同样的 input_ids 调用，永远返回同样的 logits
logits = model_fn(input_ids)   # input_ids: (B, T), logits: (B, V)
```

只看最后一个 token 的位置预测下一个 token —— `model_fn` 已经处理好这部分。

## 说明

- 输入 `input_ids` 永远是 `(1, T_init)`（单 prompt 起始）。
- 词表大小 ≤ 50，max_len ≤ 16，beam_size ≤ 4 —— 控制运行时间。
- `eos_id` 是约定的 EOS token。如果一直没出现 eos，跑满 max_len 后返回。
