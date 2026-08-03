# 解题思路：Greedy Decode

## 抽象 LM 的设计思路

为了让题目「快速可验证」，**「模型」是个函数**而非真实 LLM：

```python
def model_fn(input_ids: Tensor) -> Tensor:
    """ids (B, T) -> next-token logits (B, V)"""
```

判分时传入一个**确定性查表函数**（logits 只依赖最后一个 token），这样：
- 运行快（一次矩阵索引）
- 完全可复现（无随机）
- 但足够测出解码逻辑的对错

## Greedy Decode

最简单的解码：每步取最大概率的 token。

```python
def greedy_decode(model_fn, input_ids, max_len, eos_id):
    seq = input_ids.clone()
    for _ in range(max_len):
        logits = model_fn(seq)                              # (1, V)
        next_token = logits.argmax(dim=-1, keepdim=True)    # (1, 1)
        seq = torch.cat([seq, next_token], dim=1)
        if int(next_token.item()) == eos_id:
            break
    return seq
```

**两个边界**：
1. **遇到 EOS 立即停**（不再 append 后续 token）
2. **最长 `max_len`**（防止无限循环）

## 易错点

### `next_token.item()` 比较时类型

`argmax` 返回 `int64` tensor，`item()` 转 Python int。`eos_id` 也是 Python
int。两者用 `==` 比较是合法的（PyTorch 自动转换），但**保险写法**是显式转 int：

```python
if int(next_token.item()) == eos_id:
```

### 不要原地修改 input_ids

先 `input_ids.clone()` 再 append，避免污染调用方传入的张量。

## 进阶

Greedy 是「每步只保留 1 个最优候选」的特例。保留 `k` 个候选就是
**Beam Search**（见 `pytorch.llm.beam_search`）；引入随机性就是
**top-k / top-p sampling**（见 `pytorch.llm.top_k_top_p_sampling`）。
