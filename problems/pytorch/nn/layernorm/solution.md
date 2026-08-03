# 解题思路：LayerNorm 模块

## 一句话思路

层归一化（Layer Normalization, LayerNorm）做的事很简单：对每个样本、在最后一维上
「减均值、除标准差」，把这一层特征拉成均值 0、方差 1，再用可学习的 `weight` 和
`bias` 缩放平移。这是第一道模块题，重点是学会在 `__init__` 里用 `nn.Parameter` 建
参数、在 `forward` 里正确沿最后一维归一化。

## 从直觉到公式

### 为什么要归一化

深层网络里，各层激活的尺度可能忽大忽小，训练不稳。LayerNorm 在每个样本内部把特征
重新标准化，让后续层拿到的输入分布稳定，梯度更好走。

### 公式

给 `x`（shape `(..., D)`），沿最后一维归一化：

$$y = \frac{x - \mathrm{mean}(x)}{\sqrt{\mathrm{var}(x) + \epsilon}} \cdot \text{weight} + \text{bias}$$

`mean`、`var` 都沿最后一维算。`weight`（初始化全 1）和 `bias`（初始化全 0）是可学习
的仿射变换，让网络在归一化之后还能自己调回想要的尺度和偏移。`eps` 加在 sqrt 内部防
止方差接近 0 时除法爆炸。

### 一个关键细节：有偏方差

方差要用**有偏估计（`unbiased=False`）**，即除以 `D` 而不是 `D-1`。这和
`nn.LayerNorm` 的行为一致。

## 参考实现

```python
class LayerNorm(nn.Module):
    def __init__(self, normalized_dim, eps=1e-5):
        super().__init__()
        self.normalized_dim = normalized_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_dim))   # 初始化为 1
        self.bias = nn.Parameter(torch.zeros(normalized_dim))    # 初始化为 0

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)        # 有偏方差，除以 D
        x_norm = (x - mean) / torch.sqrt(var + self.eps)         # eps 在 sqrt 内部
        return x_norm * self.weight + self.bias
```

## 关键点

1. **必须用 `nn.Parameter`**：它把普通张量「升级」成模块参数——自动
   `requires_grad=True`、会出现在 `module.parameters()`（被优化器看到）、会进
   `state_dict`。若写成 `self.weight = torch.ones(D)`，它只是普通张量不会被训练，判分
   也会因为找不到参数而失败。

2. **`unbiased=False` 不能忘**：有偏方差除以 `D`，无偏（默认）除以 `D-1`。忘了会差一
   个 $\sqrt{(D-1)/D}$ 因子，在 `D=128` 时约 0.4% 的偏差，正好踩出 `atol=1e-5` 判分挂
   掉。

3. **`keepdim=True` 用于广播**：`mean(dim=-1, keepdim=True)` 把 `(...,D)` 归约成
   `(...,1)`，才能和原 `x` 做 `x - mean` 的广播（broadcasting）；不加 `keepdim` 维度对
   不齐会报错。

4. **`eps` 加在 sqrt 内部**：题目要求 `sqrt(var + eps)`，这和 PyTorch 一致；有些论文写
   成 `sqrt(var) + eps`（外部），数值略有差异，本题按内部。

5. **延伸**：把 LayerNorm 去掉「减均值」和「加 bias」，只保留按均方根缩放，就是现代
   LLM 常用的 RMSNorm——见 `pytorch.nn.rmsnorm`。相比沿 batch 维归一化的 BatchNorm，
   LayerNorm 与 batch 无关、训练推理行为一致，更适合序列长度多变的 Transformer。
