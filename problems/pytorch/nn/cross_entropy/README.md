# 交叉熵损失（Cross-Entropy from logits）

分类任务最核心的损失。**从 logits 直接算**（不要先手动 softmax 再 log，
要用数值稳定的 log-softmax），支持 `ignore_index`。

## 待实现函数

```python
def cross_entropy(
    logits: torch.Tensor,       # (N, C) 未归一化分数
    target: torch.Tensor,       # (N,) int64，每个元素 ∈ [0, C) 或等于 ignore_index
    ignore_index: int = -100,   # target 等于此值的样本不计入 loss
) -> torch.Tensor:              # 标量：所有有效样本的平均 loss
```

### 定义

对第 $i$ 个样本（真实类别 $y_i$）：

$$\ell_i = -\log \frac{e^{z_{i,y_i}}}{\sum_{c} e^{z_{i,c}}} = -\Bigl(z_{i,y_i} - \log\sum_c e^{z_{i,c}}\Bigr)$$

最终 loss 是**所有有效样本**（`target != ignore_index`）的 $\ell_i$ 的**平均**。

### 数值稳定

不要直接 `log(softmax(...))`——先算 `log_softmax`（内部先减 max）。可以用
`torch.logsumexp` 或自己实现：

$$\log\text{-softmax}(z)_c = z_c - \max_k z_k - \log\sum_k e^{z_k - \max_k z_k}$$

## 说明

- 输入 `logits` 是 `torch.float32`，`target` 是 `torch.int64`。
- **禁止用** `F.cross_entropy` / `F.nll_loss` / `F.log_softmax`，自己实现。
- reduction 固定为 `mean`（对有效样本平均）。
- 保证至少有一个有效样本（不会全部被 ignore）。
- 容差 `atol=1e-6`。
