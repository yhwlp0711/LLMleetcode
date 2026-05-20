# LayerNorm 模块

实现 Layer Normalization 作为 `torch.nn.Module`。这是第一道 **Pattern B**
（模块题）—— 需要在 `__init__` 里创建参数，在 `forward` 里使用它们。

## 待实现类

```python
class LayerNorm(nn.Module):
    def __init__(self, normalized_dim: int, eps: float = 1e-5):
        ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
```

## 要求

### `__init__`

创建以下两个**可学习参数**（名字必须一致 —— 判分会用 `load_state_dict`
注入参考权重）：

- `self.weight`：shape `(normalized_dim,)` 的 `nn.Parameter`，**初始化为 1**。
- `self.bias`：shape `(normalized_dim,)` 的 `nn.Parameter`，**初始化为 0**。

另外存 `self.eps` 和 `self.normalized_dim`。

### `forward`

给定 `x`（shape `(..., normalized_dim)`），在**最后一维**做归一化：

$$y = \frac{x - \mathrm{mean}(x)}{\sqrt{\mathrm{var}(x) + \epsilon}} \cdot \text{weight} + \text{bias}$$

其中 `mean` 和 `var` 沿最后一维计算，`var` 使用**有偏估计**（`unbiased=False`，
除以 `D` 而不是 `D-1`）。

## 判分

两类用例：

1. **Init 检查**：验证 `self.weight` 全为 1、`self.bias` 全为 0、shape 正确。
2. **Forward 检查**：通过 `load_state_dict` 把参考权重注入你的模块，然后
   对比 `forward(x)` 与参考实现的输出，容差 `atol=1e-5`。
