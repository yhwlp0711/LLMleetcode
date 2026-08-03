# 解题思路：数值稳定的 Sigmoid / Softmax

## 1. `sigmoid`

```python
def sigmoid(x):
    pos = x >= 0
    neg = ~pos
    out = torch.empty_like(x)
    out[pos] = 1.0 / (1.0 + torch.exp(-x[pos]))
    exp_x = torch.exp(x[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    return out
```

**为什么要分两支？** 朴素 `1/(1+exp(-x))`：
- `x = -1000` → `exp(1000)` = inf → 结果 `1/inf = 0`（碰巧对了，但中间溢出了）
- `x = 1000` → `exp(-1000)` = 0 → 结果 `1/1 = 1`（没问题）

分支法让 `exp` 的参数**永远 ≤ 0**：
- `x ≥ 0`：`exp(-x)` ∈ (0, 1]，安全
- `x < 0`：`exp(x)` ∈ (0, 1]，安全

两种写法数学等价：`e^x / (1+e^x) = 1 / (1+e^{-x})`。

## 2. `softmax`

```python
def softmax(x, dim=-1):
    m = x.max(dim=dim, keepdim=True).values
    e = (x - m).exp()
    return e / e.sum(dim=dim, keepdim=True)
```

**数值稳定性**：先减去 max 后所有指数 ≤ 0，`exp` ∈ (0, 1]，不溢出。
数学上 softmax 平移不变：`softmax(x) == softmax(x - c)` 对任意常数 `c`。

**`keepdim=True` 是关键**：让 max / sum 结果保持维度，方便广播回原 shape。

## 数值稳定性总结

sigmoid 和 softmax 的稳定性是面试高频问题。核心原则：**让 `exp` 的参数
永远不超过 0**（或者让分子分母的 exp 参数范围一致），避免 overflow。

对于 fp16，`exp(x)` 在 `x > 11` 就 overflow。工业代码通常用 `torch.sigmoid`
/ `F.softmax`（它们内部已处理好），但面试时要能手写稳定版本。
