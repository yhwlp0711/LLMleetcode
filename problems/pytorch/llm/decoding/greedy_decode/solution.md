# 解题思路：Greedy Decode

## 一句话思路

贪心解码（greedy decode）是最简单的自回归生成：每一步都取概率最大的 token
（`argmax`），拼到序列末尾，遇到结束符 `eos_id` 就停，最多生成 `max_len` 个。核心
就是一个循环，难点只在两个停止边界。

## 拆解思路

### 「模型」是一个函数

为了让题目跑得快又可复现，题目把 LLM 抽象成一个确定性函数 `model_fn`：给它当前
`input_ids`，返回下一个 token 的打分（logits）。它内部只看最后一个位置来预测下一
个 token，我们不用关心模型细节。

### 循环逻辑

从 prompt 开始，每一步：

1. 调 `model_fn(seq)` 拿 logits，形状 `(1, V)`（V 是词表大小）。
2. 取 `argmax` 作为下一个 token（贪心：只要当前最优）。
3. 把它拼到 `seq` 末尾。
4. 如果这个 token 是 `eos_id`，立刻停止。

再套一个「最多循环 `max_len` 次」的上限，防止模型一直不吐 `eos` 导致死循环。

## 参考实现

```python
import torch

def greedy_decode(model_fn, input_ids, max_len, eos_id):
    seq = input_ids.clone()                                  # 别改动传入的张量
    for _ in range(max_len):
        logits = model_fn(seq)                               # (1, V)
        next_token = logits.argmax(dim=-1, keepdim=True)     # (1, 1) 取最大
        seq = torch.cat([seq, next_token], dim=1)            # 拼到末尾
        if int(next_token.item()) == eos_id:                 # 遇到结束符停
            break
    return seq
```

## 关键点

1. **两个停止条件缺一不可**：遇到 `eos_id` 立即 `break`（且 `eos` 本身要保留在序
   列里），以及最多循环 `max_len` 次的硬上限。

2. **先 `input_ids.clone()` 再拼接**。避免原地修改调用方传入的张量，防止副作用。

3. **`argmax` 加 `keepdim=True`**，让下一个 token 保持 `(1, 1)` 形状，才能直接和
   `seq` 沿 `dim=1` 拼接。

4. **比较时显式转 int**。`argmax` 返回的是 int64 张量，`int(next_token.item())` 转
   成 Python int 再和 `eos_id` 比，最稳妥。

5. **延伸**：贪心是「每步只保留 1 个候选」的特例。保留 `k` 个候选并做长度归一化就
   是 `pytorch.llm.decoding.beam_search`；引入温度和随机筛选就是
   `pytorch.llm.decoding.top_k_top_p_sampling`。
