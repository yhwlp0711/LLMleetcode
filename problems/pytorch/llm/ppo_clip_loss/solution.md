# 解题思路：PPO 裁剪损失

## 核心思想

PPO 想「尽量按优势 A 更新 policy」，但**不让单步更新太大**。用重要性
采样比率 $r = \pi_{\text{new}}/\pi_{\text{old}}$ 衡量更新幅度，超出
$[1-\epsilon, 1+\epsilon]$ 就裁剪，切断梯度。

## 公式与参考实现

$$\mathcal{L} = -\mathbb{E}\Bigl[\min\bigl(rA,\ \text{clip}(r, 1-\epsilon, 1+\epsilon)A\bigr)\Bigr]$$

```python
def ppo_clip_loss(logratio, advantages, clip_eps=0.2):
    ratio = torch.exp(logratio)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    return -torch.min(unclipped, clipped).mean()
```

## 关键点

### 1. 为什么用 `log-ratio` 再 `exp`

`ratio = exp(log π_new - log π_old)` 比 `π_new / π_old` 数值稳定（概率相除
易下溢/上溢），且 log-prob 正好是模型直接输出的量。

### 2. `min` 的作用（悲观下界）

取 unclipped 和 clipped 的**较小值**，等价于对 objective 取「悲观估计」：

- **A > 0**（动作好，想增大概率）：`r` 涨到 `1+ε` 就被截，防止过度增大。
- **A < 0**（动作差，想减小概率）：`r` 跌到 `1-ε` 就被截，防止过度减小。

`min` 保证「往有利方向更新超过 ε 时不再给额外奖励」，但**往不利方向的惩罚
不裁剪**——这是 PPO 单调改进保证的来源。

### 3. objective 取负得到 loss

论文里 PPO 最大化 objective，代码里我们最小化 loss，所以前面加负号。

### 4. 边界

- `logratio == 0`（`r == 1`，new==old）：min 两项相等 = A，`loss = -mean(A)`。
- 这是「还没更新」时的初始 loss。

## PPO 完整 loss 里还有什么（本题不考）

- **value loss**：`(V(s) - returns)^2`，训练 critic
- **entropy bonus**：`+c·H(π)` 鼓励探索
- **GAE**：算 advantages 的方式
- 完整：`L = L_clip - c1·L_value + c2·H`

本题只考最核心、最常被单独手撕的 **clipped policy loss**。

## PPO vs DPO（面试常问）

- **PPO**：online RL，需要 reward model + 采样 rollout + value function，pipeline 复杂。
- **DPO**（见 `pytorch.llm.dpo_loss`）：把偏好直接变成监督 loss，免采样、免 reward model，更稳更简单。
