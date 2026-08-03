# 解题思路：DPO 损失

## 核心公式

DPO 把「偏好对」建模成 Bradley-Terry：chosen 应比 rejected 得分更高。
「得分」用 policy 相对 reference 的 log-ratio 表示。

$$\mathcal{L}_{\text{DPO}} = -\log\sigma\Bigl(\beta\bigl[(\log\pi_\theta^{w} - \log\pi_{\text{ref}}^{w}) - (\log\pi_\theta^{l} - \log\pi_{\text{ref}}^{l})\bigr]\Bigr)$$

## 参考实现

```python
import torch.nn.functional as F

def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    delta_chosen   = policy_chosen_logps   - ref_chosen_logps
    delta_rejected = policy_rejected_logps - ref_rejected_logps
    logits = beta * (delta_chosen - delta_rejected)
    loss = -F.logsigmoid(logits)
    return loss.mean()
```

## 关键点

### 1. 为什么要减去 reference？

`Δ = log π_θ - log π_ref` 度量的是「policy 相对初始模型改变了多少」。DPO
的推导表明：最优 policy 满足 `log(π_θ/π_ref) ∝ reward`。所以用 log-ratio
之差当作「chosen 比 rejected 好多少」的隐式 reward margin。减 reference 也
起到**正则**作用，防止 policy 跑得离初始模型太远。

### 2. 用 `logsigmoid`，别用 `log(sigmoid(x))`

`sigmoid(x)` 在 `x` 很负时 → 0，`log(0) = -inf`。`F.logsigmoid` 内部用
`logsigmoid(x) = -softplus(-x) = -log(1+e^{-x})`，全程稳定。

### 3. 边界直觉

- **`Δ_chosen == Δ_rejected`**（margin=0）：`logits=0`，`loss = -log σ(0) = log 2 ≈ 0.693`。这是「没学到偏好」的基准 loss。
- **chosen 远好于 rejected**：`logits → +∞`，`σ → 1`，`loss → 0`。
- **搞反了（rejected 更好）**：`logits < 0`，loss 快速增大。

### 4. `beta` 的作用

`beta` 控制对 reference 偏离的惩罚强度 / 偏好信号的锐度。大 `beta` 让 loss
对 margin 更敏感（更快饱和）；小 `beta` 更温和。典型取值 0.1~0.5。

## DPO vs PPO（面试常问）

- **PPO**：需要单独训练 reward model，再用 RL（采样 + clipped surrogate）优化，pipeline 长、调参难。
- **DPO**：把「reward 建模 + RL」合并成一个**分类式的监督 loss**，直接在偏好数据上训练，无需采样、无需 reward model、无需 value function。稳定、实现简单，是当前对齐的主流基线之一。
