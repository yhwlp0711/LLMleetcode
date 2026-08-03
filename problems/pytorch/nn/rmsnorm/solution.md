# 解题思路：RMSNorm 模块

## 一句话思路

RMSNorm（Root Mean Square Layer Normalization）是 LayerNorm 的「精简版」：**去掉减
均值、去掉 bias**，只按均方根（root mean square）缩放，再乘一个可学习的 `weight`。
LLaMA 等现代 LLM 都用它，因为省一半的归约（reduce）却几乎不掉效果。

## 从直觉到公式

### 和 LayerNorm 的区别

LayerNorm 要「减均值再除标准差」，RMSNorm 认为**减均值这步（re-center）其实可有可
无**，只要把向量的整体尺度归一（re-scale）就够了。于是它直接用均方根当分母：

$$\text{RMS}(x) = \sqrt{\frac{1}{D}\sum_i x_i^2 + \epsilon}$$

$$y = \frac{x}{\text{RMS}(x)} \cdot \text{weight}$$

均方根就是「所有分量平方的平均再开根」，衡量向量的整体幅度。用它去除，就把 `x` 缩
放到一个稳定的尺度。没有 bias，只有一个初始化为全 1 的 `weight`。

### 一个细节：eps 加在 sqrt 内部

`eps` 加在开方**内部**（`sqrt(mean(x²) + eps)`），和 LLaMA 实现一致，防止 `x` 全 0
时分母为 0。

## 参考实现

```python
class RMSNorm(nn.Module):
    def __init__(self, normalized_dim, eps=1e-6):
        super().__init__()
        self.normalized_dim = normalized_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_dim))   # 只有 weight，无 bias

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)  # eps 在 sqrt 内
        return (x / rms) * self.weight
```

## 关键点

1. **必须用 `nn.Parameter`**：和 LayerNorm 一样，只有包成 `nn.Parameter` 的张量才会
   被训练、进 `state_dict`、被判分识别。写成普通 `torch.ones(D)` 会判分失败。

2. **`keepdim=True` 用于广播**：`mean(dim=-1, keepdim=True)` 把 `(...,D)` 归约成
   `(...,1)`，才能和原 `x` 做 `x / rms` 的广播（broadcasting）。

3. **eps 加在 sqrt 内部**：`sqrt(mean(x²) + eps)` 避免 `x` 极小时除法不稳定，和 LLaMA
   一致；写成 `sqrt(...) + eps`（外部）数值略有差异，本题按内部。

4. **一个等价的更快写法**：把「除以 sqrt」换成「乘以 rsqrt」，
   `x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * self.weight`。`rsqrt` 是倒
   数平方根，硬件有专门指令，少一次除法、数值等价，判分同样能过。

5. **延伸**：RMSNorm 是在 LayerNorm 基础上「砍掉冗余设计」的典型——见
   `pytorch.nn.layernorm` 对比两者。这种精简换速度的思路在现代 LLM 里很常见（比如前馈
   网络换成门控的 `pytorch.nn.gated_activations`）。
