# SwiGLU FFN（LLaMA 风格）

实现 LLaMA / Mistral / Mixtral / Gemma 使用的位置无关 FFN。和原始 Transformer
的 `Linear → ReLU → Linear`（2 个矩阵）不同，现代 LLM 用**门控** FFN，含
**3 个**线性层：

$$\text{FFN}(x) = W_{\text{down}}\bigl(\text{SiLU}(W_{\text{gate}}\,x) \odot W_{\text{up}}\,x\bigr)$$

## 待实现类

```python
class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
```

## 要求

### `__init__`

创建三个 Linear 层，名字必须一致（判分会用 `load_state_dict` 注入参考权
重）：

- `self.gate_proj`：`nn.Linear(d_model, d_ff, bias=False)`
- `self.up_proj`：  `nn.Linear(d_model, d_ff, bias=False)`
- `self.down_proj`：`nn.Linear(d_ff, d_model, bias=False)`

**都没有 bias**（与 LLaMA 一致）。不需要自定义 init —— 判分会用参考权重
覆盖你的随机初始化。

### `forward`

给定 `x`（shape `(..., d_model)`）：

```
gate = self.gate_proj(x)         # (..., d_ff)
up   = self.up_proj(x)           # (..., d_ff)
y    = silu(gate) * up           # (..., d_ff)  逐元素相乘
out  = self.down_proj(y)         # (..., d_model)
```

`silu(z) = z * sigmoid(z)`。可以用 `F.silu`。

## 判分

1. **Init 检查**：三个 Linear 层都存在、shape 正确、`bias=False`。纯结构
   性检查（不对比数值，因为 `nn.Linear` 默认随机初始化）。
2. **Forward 检查**：通过 `load_state_dict` 同步参考权重，再数值对比
   （`atol=1e-5`）。

## 说明

- 不要加 dropout —— 保持 eval 行为。（参见 `docs/AUTHORING.md → Pitfalls`
  解释为什么）。
- `d_ff` 在真实 LLaMA 里约等于 `2.67 * d_model`（因为门控 FFN 有 3 个矩阵
  而非 2 个，缩小 hidden dim 才能保持参数量接近）。判分会用不同比例的
  `d_ff` 测试。
