# BCE 损失（Binary Cross-Entropy with logits）

二分类 / 多标签任务的核心损失。**从 logits 直接算**（数值稳定，不要先
sigmoid 再 log）。

## 待实现函数

```python
def bce_with_logits(
    logits: torch.Tensor,   # 任意 shape 的未归一化分数
    target: torch.Tensor,   # 同 shape，取值 ∈ {0, 1}（float）
) -> torch.Tensor:          # 标量：所有元素的平均 loss
```

### 定义

对每个元素（logit $z$，标签 $y \in \{0,1\}$）：

$$\ell = -\bigl[y \cdot \log\sigma(z) + (1-y)\cdot\log(1-\sigma(z))\bigr]$$

最终 loss 是所有元素的**平均**（对整个张量 mean）。

### 数值稳定形式

直接用 $\log\sigma(z)$ 在 $z$ 很负时会下溢。等价的稳定写法：

$$\ell = \max(z, 0) - z\cdot y + \log\bigl(1 + e^{-|z|}\bigr)$$

（这是 PyTorch `binary_cross_entropy_with_logits` 内部用的形式。）

## 说明

- 输入 `logits` 与 `target` 同 shape，都是 `torch.float32`。
- **禁止用** `F.binary_cross_entropy_with_logits` / `F.binary_cross_entropy` /
  `torch.sigmoid` / `F.logsigmoid`，自己按稳定公式实现。
- reduction 固定为 `mean`（对所有元素平均）。
- 会用极端 logits（如 `±100`）测数值稳定性。
- 容差 `atol=1e-6`。
