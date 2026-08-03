# 解题思路：数值稳定的 Sigmoid / Softmax

## 一句话思路

Sigmoid 和 Softmax 的公式都很简单，难点全在**数值稳定（numerical stability）**——
直接照公式写，遇到很大或很小的输入时 `exp` 会溢出（overflow）变成 `inf`。
核心技巧只有一个：**想办法让 `exp` 的指数永远 ≤ 0**。

## 1. Sigmoid

### 为什么不能直接照公式写？

课本公式是 $\sigma(x) = \dfrac{1}{1 + e^{-x}}$。直接翻译成 `1/(1+exp(-x))`，
在极端输入时会出问题：

- `x = -1000` → 要算 `exp(1000)`，超出浮点数能表示的范围 → 变成 `inf`。
  （结果 `1/inf = 0` 碰巧是对的，但中间已经溢出，换个场景就崩了。）
- `x = 1000` → `exp(-1000)` 下溢（underflow）成 0 → `1/1 = 1`，这支没问题。

问题出在「当 `x` 很负时，`exp(-x)` 会爆炸」。

### 解决办法：按符号分两支

关键观察：`exp` 只要指数 ≤ 0 就安全（结果落在 (0, 1] 之间）。所以按 `x` 的
正负分别用两个**数学等价**的写法，保证喂给 `exp` 的永远是负数：

- `x ≥ 0`：用 $\dfrac{1}{1+e^{-x}}$，指数 `-x ≤ 0` ✅
- `x < 0`：用 $\dfrac{e^{x}}{1+e^{x}}$，指数 `x < 0` ✅

两式相等（分子分母同乘 $e^x$ 即可互相转换）。

```python
def sigmoid(x):
    pos = x >= 0
    neg = ~pos
    out = torch.empty_like(x)
    out[pos] = 1.0 / (1.0 + torch.exp(-x[pos]))   # x≥0：指数 -x≤0
    exp_x = torch.exp(x[neg])                       # x<0：指数 x<0
    out[neg] = exp_x / (1.0 + exp_x)
    return out
```

## 2. Softmax

### 平移不变性（shift invariance）是关键

Softmax 有个好性质：给所有输入**同时加减一个常数，结果不变**：

$$\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}} = \frac{e^{x_i - c}}{\sum_j e^{x_j - c}}$$

（分子分母同时除以 $e^c$。）

利用这点，取 `c = max(x)`，那么每个 `x_i - c ≤ 0`，`exp` 就永远不会溢出。

```python
def softmax(x, dim=-1):
    m = x.max(dim=dim, keepdim=True).values   # 每行的最大值
    e = (x - m).exp()                          # 减 max 后，指数都 ≤ 0
    return e / e.sum(dim=dim, keepdim=True)
```

## 关键点

1. **核心原则：让 `exp` 的指数不超过 0**。sigmoid 靠分支、softmax 靠减去
   max，本质是同一招——把可能爆炸的指数「拉」到安全区。

2. **`keepdim=True` 不能少**。`max` / `sum` 沿某个维度归约（reduce）后那个
   维度会消失，加 `keepdim=True` 让它保留成长度 1，才能和原张量做广播
   （broadcasting）相减/相除。

3. **溢出到底多容易发生？** 对半精度（fp16），`exp(x)` 在 `x > 11` 左右就
   溢出了；训练大模型时 logits 很容易超过这个范围，所以稳定写法是刚需。

4. **延伸**：工业代码直接用 `torch.sigmoid` / `torch.softmax`，它们内部
   已经做好了同样的稳定处理。手写的意义在于理解「为什么要减 max」——这个
   技巧在交叉熵（cross-entropy）、logsumexp 里反复出现。
