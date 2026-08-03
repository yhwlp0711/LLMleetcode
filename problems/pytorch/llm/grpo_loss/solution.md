# 解题思路：GRPO 损失

## 核心思想

GRPO（Group Relative Policy Optimization，DeepSeek）是 PPO 的**去 critic** 版本。
PPO 需要一个 value 网络估 baseline 来算 advantage；GRPO 换成：对同一 prompt
采样一组回答，用**组内 reward 的 z-score**当 advantage —— 不需要 critic。

## 步骤与参考实现

```python
def grpo_loss(logratio, rewards, clip_eps=0.2, eps_std=1e-4):
    # 1. 组内优势归一化（GRPO 的关键）
    adv = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + eps_std)

    # 2. PPO 裁剪目标（和 ppo_clip_loss 一样）
    ratio = torch.exp(logratio)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    return -torch.min(unclipped, clipped).mean()
```

## 关键点

### 1. 组内 z-score 归一化

$$A_i = \frac{r_i - \operatorname{mean}(r)}{\operatorname{std}(r) + \epsilon}$$

- **减 mean**：比组内平均好的回答 A>0（增大概率），差的 A<0（减小概率）。
  这就是「相对」的含义——不看绝对 reward，看组内排名。
- **除 std**：把不同 prompt / 不同难度的 reward 尺度归一化，训练更稳。
- **std 用总体标准差**（`unbiased=False`，分母 N 不是 N−1）——对齐 DeepSeek / trl。
- `+eps` 防止组内 reward 全相等时除零。

### 2. 后半段就是 PPO

优势算好后，裁剪目标和 `pytorch.llm.ppo_clip_loss` 完全一样。所以 GRPO ≈
「组归一化优势」+「PPO clip」。

### 3. 边界直觉

- **logratio == 0**（还没更新）：`loss = -mean(A)`，而组归一化后 `mean(A) ≈ 0`，
  所以初始 loss ≈ 0。
- 组内 reward 全相同：`std ≈ 0`，A ≈ 0，loss ≈ 0（没有可学的相对信号）。

## GRPO vs PPO vs DPO

| | critic | 采样 | reward model |
|---|---|---|---|
| **PPO** | 需要 value 网络 | rollout | 需要 |
| **GRPO** | **不需要**（组内 baseline） | 每 prompt 采一组 | 需要 |
| **DPO** | 不需要 | 不需要 | **不需要**（直接用偏好对） |

GRPO 省掉了 critic，显存和实现都更省，是 DeepSeek-R1 等推理模型 RL 训练的主力。
