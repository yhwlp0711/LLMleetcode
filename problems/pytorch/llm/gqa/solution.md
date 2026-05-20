# 解题思路：Grouped-Query Attention (GQA)

## GQA 是什么？

MHA 里 Q/K/V 各有 `H` 个头。GQA 让 K/V **少一些头**，多个 Q head 共享同
一组 K/V：

```
MHA:  H q-heads, H kv-heads      (1:1)
GQA:  H q-heads, G kv-heads      (H:G, G < H, G | H)
MQA:  H q-heads, 1 kv-head       (G=1，极端情况)
```

**节省什么？** —— KV cache 内存。GQA 把 KV cache 大小缩到 `G/H` 倍。对
长上下文 LLM 推理是巨大优化（KV cache 经常占大头）。

LLaMA-2 70B 用 GQA：64 q-heads, 8 kv-heads（H:G=8）。

## 参考实现

```python
def gqa(x, W_q, W_k, W_v, W_o, num_q_heads, num_kv_heads, mask=None):
    B, T, D = x.shape
    repeats = num_q_heads // num_kv_heads
    head_dim = W_q.shape[1] // num_q_heads

    # 投影
    q = x @ W_q
    k = x @ W_k
    v = x @ W_v

    # 切头
    q = q.reshape(B, T, num_q_heads, head_dim).transpose(1, 2)
    k = k.reshape(B, T, num_kv_heads, head_dim).transpose(1, 2)
    v = v.reshape(B, T, num_kv_heads, head_dim).transpose(1, 2)

    # 把 K/V 头数扩到 num_q_heads
    k = k.repeat_interleave(repeats, dim=1)
    v = v.repeat_interleave(repeats, dim=1)

    # 后面就是标准 SDPA + 合头
    scores = q @ k.transpose(-2, -1) / sqrt(head_dim)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    out = attn @ v
    out = out.transpose(1, 2).reshape(B, T, num_q_heads * head_dim)
    return out @ W_o
```

## 核心新操作：`repeat_interleave`

把 K/V 从 `(B, num_kv_heads, T, head_dim)` 扩到
`(B, num_q_heads, T, head_dim)`，让它们能和 Q 做注意力。

```python
k = k.repeat_interleave(repeats, dim=1)
```

`repeat_interleave(repeats=2, dim=1)` 在 dim=1 把每个元素重复 2 次：

```
原: [a, b, c]
重复后: [a, a, b, b, c, c]
```

跟 `repeat` 不一样（后者是 `[a, b, c, a, b, c]`）。**这道题必须用
interleave**，因为 Q 的 head 是连续分组的：q-heads 0,1 共享 kv-head 0；
q-heads 2,3 共享 kv-head 1；...

## 关键点

### 1. 形状对账

| Tensor | 形状 |
|---|---|
| `q` (切头后) | `(B, num_q_heads, T, head_dim)` |
| `k` / `v` (切头后) | `(B, num_kv_heads, T, head_dim)` |
| `k` / `v` (重复后) | `(B, num_q_heads, T, head_dim)` |
| `scores` | `(B, num_q_heads, T, T)` |
| `out` (合头前) | `(B, num_q_heads, T, head_dim)` |
| `out` (最终) | `(B, T, D)` |

### 2. `W_k` / `W_v` 的列数较小

`W_k` shape 是 `(D, num_kv_heads * head_dim)`，比 `W_q` 的 `(D, num_q_heads
* head_dim)` 窄。**这是 GQA 节省参数的来源**：投影矩阵小了，K/V cache 也
小了。

### 3. `repeats = num_q_heads // num_kv_heads`

必须**整除**。LLaMA 系列模型的常见配置 `H:G = 8:1`，所以 repeats = 8。

### 4. 退化情形不要写特判

- `num_kv_heads == num_q_heads`：`repeats=1`，`repeat_interleave(1)` 是
  no-op，自动等价于 MHA。
- `num_kv_heads == 1`：`repeats = num_q_heads`，所有 Q head 共享同一对
  K/V，自动等价于 MQA。

**统一代码可以同时处理三种**，不需要 `if-else`。这是优雅实现的标志。

## 跟 KV Cache 的协同

GQA + KV cache 是**最大化推理节省**的组合：

- KV cache 大小：`B × T × G × head_dim × 2`（vs MHA 的 `× H`）
- 当 H=64, G=8 时，KV cache 只有 MHA 的 1/8

LLaMA-3 70B 在 32k context 下 KV cache 大小：MHA 会爆显存，GQA 才能塞下。

## 易错点

### 1. 写成 `repeat` 而不是 `repeat_interleave`

`k.repeat(repeats, ...)` 会让 K head 排列变成 `[kv0, kv1, ..., kv0, kv1,
...]`（按组循环），不符合「q-head 连续分组对应 kv-head」的语义。结果数
值会错。

### 2. 忘了 head_dim 怎么算

`head_dim = W_q.shape[1] // num_q_heads`，**不是** `D / num_q_heads`。
虽然在标准配置下两者相等，但有些模型让 Q 投影到不同的维度（比如 head_dim
独立于 D）。从 `W_q` 推导更鲁棒。

### 3. mask shape 仍然是 num_q_heads

mask 跟 attention scores 同 shape `(B, num_q_heads, T, T)`。**不要**给
mask 用 num_kv_heads —— K/V 已经被 repeat 到 num_q_heads，mask 也是
num_q_heads 维度。
