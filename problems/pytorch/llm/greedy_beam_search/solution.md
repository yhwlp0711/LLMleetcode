# 解题思路：Greedy Decode 与 Beam Search

## 抽象 LM 的设计思路

为了让题目「快速可验证」，**「模型」是个函数**而非真实 LLM：

```python
def model_fn(input_ids: Tensor) -> Tensor:
    """ids (B, T) -> next-token logits (B, V)"""
```

判分时传入一个**确定性查表函数**（logits 只依赖最后一个 token），这样
:
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

## Beam Search

Beam search 同时维护 `k` 个候选序列，每步扩展再剪枝。

### 算法骨架

```
初始化:
    用 prompt 跑一次 → 取 top-k 作为初始 beams
    每个 beam 的 score = log_prob(token)

每步:
    1. 对所有 k 个 beams 各调用 model_fn → (k, V) logits
    2. 计算累积 score: cand_scores[i, j] = scores[i] + log_p[i, j]
    3. flatten 到 (k * V,)，取 top-k → 得到 (beam_idx, token_idx) 对
    4. 用 beam_idx 重排现有 beams，append token_idx
    5. 更新 finished mask（碰到 eos 的 beam）

终止:
    所有 beams finished 或 步数到 max_len
返回:
    用「长度归一化 score」选最优 beam
```

### 关键技巧

#### 1. `flat.topk(k)` + 整数除法/取模 拆解 (beam, token)

```python
flat = cand_scores.view(-1)                # (k * V,)
topv, topi = flat.topk(beam_size)
beam_idx = topi // V                        # 哪个原 beam
token_idx = topi % V                        # 选了哪个 token
```

这是 beam search 最经典的 trick —— **不显式构造 (k, V) 索引网格，靠
flatten + 除法还原**。代码 3 行，全向量化。

#### 2. 已结束的 beam 怎么处理？

要避免它们被「再次选中并 append 错误 token」。两个常见做法：

**做法 A（本题用的）**：把已 finished beam 的 logits 改成「eos=0, 其他=-inf」，
这样它们「只能继续选 eos」，score 不变。

```python
masked_log_p = log_p.clone()
if finished.any():
    masked_log_p[finished] = float("-inf")
    masked_log_p[finished, eos_id] = 0.0
```

这样它们的 cand_scores 等于 (旧 score + 0) = 旧 score；token 永远是 eos。
活的 beam 不受影响。

**做法 B**：把 finished beam 从候选池里完全排除，单独维护「已完成」列表，
结束时合并。代码更复杂，但是某些 HF 实现方式。

#### 3. 长度归一化

```python
norm_scores = scores / lengths.float().clamp(min=1)
best = norm_scores.argmax()
```

**为什么需要？** 长度长的序列累积 log-prob 自然更小（每步都减去一个正数），
直接 argmax 偏好短序列。除以 length 后比较「平均每 token log-prob」，更公平。

更高级的做法是 length penalty `length^alpha`（alpha < 1），但本题简化用
`length^1`。

## 易错点

### 1. `next_token.item()` 比较时类型

`argmax` 返回 `int64` tensor，`item()` 转 Python int。`eos_id` 也是 Python
int。两者用 `==` 比较是合法的（PyTorch 自动转换），但**保险写法**是显式
转 int：

```python
if int(next_token.item()) == eos_id:
```

### 2. 初始 beam 也可能是 eos

如果第一步取的 top-k 里有 eos token，对应 beam 一开始就是 finished。要在
初始化时也更新 `finished` mask，不然下一步它还会被错误扩展。

### 3. `lengths` 的累加规则

只有「这一步实际增加了新 token」（即 beam 没 finished）的才 `length += 1`。
已 finished 的 beam length 不变。

```python
lengths = lengths[beam_idx] + (~was_finished).long()
```

这是把所有逻辑揉到一个向量化表达里的精髓。

## 为什么 Beam Search 仍然有人考？

虽然 LLM 推理时主流是 sampling（greedy/top-k/top-p），但：

- **机器翻译 / Seq2Seq 任务**：beam search 仍然是标准做法
- **长答案生成**：beam 能避免局部贪心导致的低质量结尾
- **面试观察点**：beam search 把「调度逻辑」考得很全 —— 队列管理、向量
  化、边界处理，是综合能力测试题

## 性能

每步 O(k · V) for top-k；总 O(max_len · k · V)。对小 V/k 极快。
真实 LLM（V = 50k+）会用更聪明的剪枝（GPU 友好的 topk on flat tensor）。
