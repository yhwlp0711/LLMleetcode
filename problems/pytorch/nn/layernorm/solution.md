# 解题思路：LayerNorm 模块

## 参考实现

```python
class LayerNorm(nn.Module):
    def __init__(self, normalized_dim, eps=1e-5):
        super().__init__()
        self.normalized_dim = normalized_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_dim))
        self.bias = nn.Parameter(torch.zeros(normalized_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.weight + self.bias
```

## 关键点

### 1. `nn.Parameter` vs 普通张量

`nn.Parameter(t)` 把一个张量「升级」成模块参数：

- 自动 `requires_grad=True`
- 会出现在 `module.parameters()` 里（被优化器看到）
- `module.state_dict()` 会包含它

如果你写 `self.weight = torch.ones(D)`（**不**包 Parameter），它只是一个普
通张量，不会被训练，judge 也会因为 `assert_param_names` 检查不到而失败。

### 2. `unbiased=False` —— 有偏方差

`torch.var(x, unbiased=False)` 除以 `N`，`unbiased=True`（默认）除以 `N-1`。
LayerNorm 论文用 `N`（即有偏估计），跟 `nn.LayerNorm` 一致。

如果忘掉 `unbiased=False`，结果会差一个 $\sqrt{(N-1)/N}$ 的因子 —— 在
`normalized_dim=128` 时差距 ~0.4%，正好踩在 `atol=1e-5` 之外，判分挂掉。

### 3. `keepdim=True`

`mean(dim=-1, keepdim=True)`：shape `(..., D) → (..., 1)`，方便后续广播。
没 `keepdim`：`(..., D) → (...,)`，做 `x - mean` 时维度对不齐报错。

### 4. `eps` 加在哪？

题目要求 `sqrt(var + eps)`，eps 在 sqrt **内部**。这是 PyTorch / 大多数实
现的写法。**有些论文**把 eps 加在 sqrt **外部**（`sqrt(var) + eps`），结
果略有差异。本题按内部加。

## 为什么 LayerNorm 比 BatchNorm 在 Transformer 里更好？

- **BatchNorm** 沿 batch 维归一化，依赖 batch 内统计；推理时要维护
  running mean/var。在 NLP 里 batch 内序列长度差异大，统计噪声大。
- **LayerNorm** 沿 feature 维（最后一维）归一化，**与 batch 无关**；训练
  和推理行为一致，不需要 running stats。Transformer 全栈 LayerNorm。

## 测试 fixture 的小心机

判分时会把 `weight` 和 `bias` 替换成随机非 1/0 值（用 `sync_weights`），
然后对比 forward。这能区分「你忘了乘 weight / 加 bias」和「数学正确但参
数没用上」这两种 bug。
