# 解题思路：PyTorch 张量操作热身

## 1. `flatten_and_concat`

`reshape(-1)` 把任意形状压成一维；`torch.cat` 拼接，注意 `dim=0`：

```python
def flatten_and_concat(a, b):
    return torch.cat([a.reshape(-1), b.reshape(-1)])
```

> `view(-1)` 也行，但要求 contiguous；`reshape(-1)` 不要求，更鲁棒。

## 2. `row_softmax` —— 数值稳定

朴素 softmax：`exp(x) / exp(x).sum()`。当 `x` 里有大数（比如 1000），
`exp(1000) = inf`，结果直接崩。

**减去每行最大值不改变结果**（softmax 平移不变）：

```python
def row_softmax(x):
    m = x.max(dim=-1, keepdim=True).values     # (N, 1)
    e = (x - m).exp()                          # 全部 ≤ 0，exp ∈ (0, 1]
    return e / e.sum(dim=-1, keepdim=True)
```

`keepdim=True` 是为了广播：`(N, D) - (N, 1)`、`(N, D) / (N, 1)`。

## 3. `pairwise_squared_distance`

数学技巧：

$$\|x\_i - y\_j\|^2 = \|x\_i\|^2 - 2 x\_i^\top y\_j + \|y\_j\|^2$$

向量化实现：

```python
def pairwise_squared_distance(x, y):
    x2 = (x * x).sum(dim=-1, keepdim=True)              # (N, 1)
    y2 = (y * y).sum(dim=-1, keepdim=True).transpose(0, 1)  # (1, M)
    xy = x @ y.transpose(0, 1)                          # (N, M)
    return (x2 - 2 * xy + y2).clamp(min=0.0)
```

最后 `clamp(min=0)` 是因为浮点误差可能让小距离算出负数（理论上 ≥ 0）。

**替代写法**（用广播，代码更短但内存更大）：

```python
return ((x[:, None, :] - y[None, :, :]) ** 2).sum(-1)
# 中间张量 (N, M, D)，N、M、D 大时容易爆内存
```

## 4. `masked_mean`

把 mask 转 float、扩一维方便广播，求和后除以 mask 的总和：

```python
def masked_mean(x, mask):
    m = mask.to(x.dtype).unsqueeze(-1)              # (B, T, 1)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
```

- `mask.to(x.dtype)`：bool → float，方便参与算术。
- `unsqueeze(-1)`：让 mask 能与 `x` 的最后一维 broadcast。
- `clamp(min=1.0)`：保险起见，如果某 batch 全是 False 也不会除 0（虽然题
  目保证不会发生）。

## 5. `top_k_indices`

`torch.topk` 已经默认 `sorted=True`，并且 CPU 上的实现是稳定的（并列时索
引小的在前）：

```python
def top_k_indices(scores, k):
    _, idx = torch.topk(scores, k=k, largest=True, sorted=True)
    return idx
```

注意 `topk` 返回的是 `(values, indices)` 命名元组，只取第二个。如果是 GPU
上的实现，并列时索引顺序可能不稳定，需要更复杂的处理（先排序再取前 k）。
本题在 CPU 上跑，所以直接 `topk` 即可。
