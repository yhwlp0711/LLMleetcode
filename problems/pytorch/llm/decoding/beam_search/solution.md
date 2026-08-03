# 解题思路：Beam Search

## 一句话思路

集束搜索（beam search）同时维护 `beam_size` 个候选序列，每步把它们各自扩展到所有
可能的下一个 token，按**累积 log 概率**排序，只保留最好的 `beam_size` 个，其余剪
枝。核心是「扩展 → 打平 → 取 top-k → 还原是哪个 beam 选了哪个 token」这套向量化操
作。

## 拆解思路

### 「模型」是一个函数

和 `pytorch.llm.decoding.greedy_decode` 一样，题目把 LLM 抽象成确定性的
`model_fn`：喂 `input_ids`，返回下一个 token 的 logits。快、可复现，又能测出解码逻
辑对不对。

### 为什么用累积 log 概率

一个序列的概率是每步概率连乘。概率相乘容易下溢（underflow），所以取对数把连乘变成
连加：序列 score = 每步 `log_softmax` 之和。贪心只顾当下最优，beam search 保留多个
候选，能在后面「反悔」，找到整体概率更高的序列。

### 每步在做什么

1. 对当前 `beam_size` 个 beam 各调一次 `model_fn`，得到 `(beam_size, V)` 的
   `log_softmax`。
2. 累加：`候选score[i, j] = 旧score[i] + log_p[i, j]`，形状 `(beam_size, V)`。
3. 把它**打平**成 `(beam_size * V,)`，取全局 top-`beam_size`。
4. 用整数除法/取模还原：`beam_idx = topi // V`（来自哪个旧 beam）、
   `token_idx = topi % V`（选了哪个 token）。
5. 用 `beam_idx` 重排 beam 并拼上 `token_idx`。

### 已结束的 beam 怎么办

某个 beam 生成了 `eos` 就算结束，不该再扩展。技巧：把已结束 beam 的下一步 logits
改成「eos 位置 = 0、其余 = $-\infty$」，这样它只能「继续选 eos」、score 保持不变，
既不干扰活着的 beam，也不会被错误延长。

### 长度归一化

累积 log 概率每步都在减小，直接比会偏袒短序列。所以用 `score / length`（平均每
token 的 log 概率）来挑最终最优 beam，更公平。

## 参考实现

```python
import torch
import torch.nn.functional as F

def beam_search(model_fn, input_ids, max_len, beam_size, eos_id):
    # 初始化：用 prompt 跑一次，取 top-k 当初始 beams
    logits = model_fn(input_ids)                      # (1, V)
    log_probs = F.log_softmax(logits[0], dim=-1)      # (V,)
    topv, topi = log_probs.topk(beam_size)
    beams = torch.cat([input_ids.expand(beam_size, -1), topi.unsqueeze(1)], dim=1)
    scores = topv                                     # 累积 log-prob
    finished = topi == eos_id                         # 初始就可能是 eos
    lengths = torch.ones(beam_size, dtype=torch.long)

    for step in range(1, max_len):
        if finished.all():
            break
        log_p = F.log_softmax(model_fn(beams), dim=-1)  # (K, V)
        V = log_p.shape[-1]

        masked_log_p = log_p.clone()                  # 已结束 beam：只能续 eos
        if finished.any():
            masked_log_p[finished] = float("-inf")
            masked_log_p[finished, eos_id] = 0.0

        cand = scores.unsqueeze(1) + masked_log_p     # (K, V) 累积候选
        topv, topi = cand.view(-1).topk(beam_size)    # 打平取全局 top-k
        beam_idx = topi // V                          # 来自哪个 beam
        token_idx = topi % V                          # 选了哪个 token

        beams = torch.cat([beams[beam_idx], token_idx.unsqueeze(1)], dim=1)
        scores = topv

        was_finished = finished[beam_idx]
        finished = was_finished | ((token_idx == eos_id) & ~was_finished)
        lengths = lengths[beam_idx] + (~was_finished).long()  # 活着才 +1

    norm_scores = scores / lengths.float().clamp(min=1)       # 长度归一化
    return beams[norm_scores.argmax() : norm_scores.argmax() + 1]
```

## 关键点

1. **打平 + 除法/取模是 beam search 的核心 trick**。把 `(beam_size, V)` 候选拉成一
   维取 top-k，再用 `topi // V` 和 `topi % V` 还原出「哪个 beam、哪个 token」，全
   程向量化，不用手写索引网格。

2. **累积 score 用 log 概率相加**。`F.log_softmax` 把每步概率取对数，序列 score 就
   是逐步累加，避免概率连乘的下溢。

3. **已结束 beam 冻结处理**。把它们的下一步 logits 设成「eos=0、其余 $-\infty$」，
   score 不变、只能续 eos，从而不打扰活着的 beam。初始 top-k 里若已有 eos，也要在
   初始化时标记 `finished`。

4. **`lengths` 只对「实际新增 token」的 beam 累加**：`lengths[beam_idx] +
   (~was_finished)`。已结束的 beam 长度不再增长，长度归一化才公平。

5. **重排 beam 用 `beams[beam_idx]`**。top-k 可能让多个新 beam 来自同一个旧 beam，
   用 `beam_idx` 做花式索引正好复制/丢弃对应的历史。

6. **延伸**：`beam_size == 1` 时 beam search 退化成 `pytorch.llm.decoding.greedy_decode`。
   想要多样性、带随机的解码则用 `pytorch.llm.decoding.top_k_top_p_sampling`。
