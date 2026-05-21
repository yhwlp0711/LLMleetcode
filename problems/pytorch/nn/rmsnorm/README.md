# RMSNorm 模块

实现 Root Mean Square Layer Normalization（RMSNorm），LLaMA 等现代 LLM 使
用的归一化方式。它是 LayerNorm 的简化：**去掉均值减法**和**偏置加法**。

## 待实现类

```python
class RMSNorm(nn.Module):
    def __init__(self, normalized_dim: int, eps: float = 1e-6):
        ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
```

## 要求

### `__init__`

- `self.weight`：`(normalized_dim,)` 的 `nn.Parameter`，**初始化为 1**。
  （没有 bias。）
- 存 `self.eps` 和 `self.normalized_dim`。

### `forward`

给定 `x`（shape `(..., normalized_dim)`）：

$$\text{RMS}(x) = \sqrt{\frac{1}{D}\sum\_i x\_i^2 + \epsilon}$$

$$y = \frac{x}{\text{RMS}(x)} \cdot \text{weight}$$

**注意**：`eps` 加在 sqrt **内部**（与 LLaMA 实现一致）。

## 判分

- **Init**：`weight` 存在、shape 正确、初始为全 1。
- **Forward**：通过 `load_state_dict` 注入参考权重，再做数值对比
  （`atol=1e-5`）。
