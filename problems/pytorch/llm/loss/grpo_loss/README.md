# GRPO 损失（Group Relative Policy Optimization）

GRPO 是 DeepSeek 提出的 PPO 变体，**去掉了 value/critic 网络**：对同一个 prompt
采样一组（group）回答，用**组内 reward 的相对好坏**当作优势（advantage），再套
PPO 的裁剪目标。本题实现 loss 前向：从「组内 rewards + log-ratio」算出 loss。

## 待实现函数

```python
def grpo_loss(
    logratio: torch.Tensor,   # (G,) = log π_new - log π_old（组内 G 个样本）
    rewards: torch.Tensor,    # (G,) 组内每个回答的标量 reward
    clip_eps: float = 0.2,
    eps_std: float = 1e-4,    # 标准化时防止除零
) -> torch.Tensor:            # 标量：组内平均 loss
```

### 步骤

1. **组内优势归一化**（GRPO 的核心）：

$$A_i = \frac{r_i - \operatorname{mean}(r)}{\operatorname{std}(r) + \epsilon_{\text{std}}}$$

其中 mean / std 都在这一组 G 个 reward 上计算（std 用**总体标准差**，即 `unbiased=False`）。

2. **PPO 裁剪目标**（和 `ppo_clip_loss` 一致）：

$$r_i^{\text{ratio}} = \exp(\text{logratio}_i), \qquad
\mathcal{L} = -\operatorname{mean}_i\Bigl[\min\bigl(r_i^{\text{ratio}} A_i,\ \text{clip}(r_i^{\text{ratio}}, 1-\epsilon, 1+\epsilon) A_i\bigr)\Bigr]$$

## 说明

- `logratio` 与 `rewards` 都是 shape `(G,)` 的 `torch.float32`（一个 group）。
- std 用**总体标准差** `.std(unbiased=False)`（对齐 DeepSeek / trl 实现）。
- 分母加 `eps_std` 防止组内 reward 全相同时除零。
- 裁剪与 reduction 同 PPO：`clamp(ratio, 1±eps)`、batch **mean**。
- 容差 `atol=1e-6`。

> GRPO 和 PPO 的关键区别：advantage 来自**组内 z-score 归一化的 reward**，
> 而非 GAE / critic 估计的 value。裁剪部分见 `pytorch.llm.loss.ppo_clip_loss` 对比。
