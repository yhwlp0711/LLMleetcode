# 解题思路：RMSNorm

## 参考实现

```python
class RMSNorm(nn.Module):
    def __init__(self, normalized_dim, eps=1e-6):
        super().__init__()
        self.normalized_dim = normalized_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_dim))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight
```

## RMSNorm vs LayerNorm

| | LayerNorm | RMSNorm |
|---|---|---|
| 减均值 | ✓ | ✗ |
| 除以 std | ✓ | ✗（除以 RMS） |
| weight | ✓ | ✓ |
| bias | ✓ | ✗ |
| 计算量 | 2 次 reduce + 1 次 sqrt | 1 次 reduce + 1 次 sqrt |

RMSNorm 的核心论点：减均值（re-center）对效果几乎没影响，只要 re-scale 就
够了，省一半的 reduce。LLaMA 系列实测训练稳定且更快，所以推广开来。

## 实现细节

### 1. `x.pow(2)` vs `x ** 2` vs `x * x`

三者数学等价、性能基本一致。`x.pow(2)` 最明确表达「平方」语义。本题任选。

### 2. `eps` 加在 sqrt 内 vs 外

- **内**（本题）：`sqrt(rms² + eps)`，避免 `rms` 极小时除法不稳定。
- **外**：`sqrt(rms²) + eps`，等价于在分母上加一个常数。

两种都有论文用过，差异在 `rms` 接近 0 时才显现。LLaMA 用内部，本题按
LLaMA。

### 3. 等价写法：`rsqrt`

`rsqrt(x) = 1 / sqrt(x)` 在底层有专门的快速指令（hardware reciprocal
sqrt）：

```python
def forward(self, x):
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight
```

把「除以 sqrt」改成「乘以 rsqrt」，少一个除法，理论上更快（数值上等价，
本题判分能过）。

## 为什么 LayerNorm 有 bias 而 RMSNorm 没？

RMSNorm 论文实验显示 bias 对效果没什么帮助，但增加参数量和带宽。LLaMA 等
工程导向的模型直接砍掉。这种「砍多余设计」的风格在现代 LLM 里很常见。
