# 解题思路：GRPO 损失

## 一句话思路

GRPO（Group Relative Policy Optimization，组相对策略优化）是 DeepSeek 提出的 PPO
变体：去掉价值网络（value network），改用**组内 reward 的 z-score 归一化**当作优势
（advantage），再套 PPO 的裁剪目标。所以实现分两步：组内归一化算优势 → PPO 裁剪。

## 从直觉到公式

### 组内归一化：为什么？

对同一个 prompt，模型采样了一组（group）回答，每条都有个标量 reward。我们不看
reward 的绝对值，而是看**组内谁比平均好、谁比平均差**——比组内平均好的回答应该增大
概率（$A_i > 0$），差的应该减小概率（$A_i < 0$）。这就是「group relative」的意
思。

做法就是 z-score 归一化：

$$A_i = \frac{r_i - \operatorname{mean}(r)}{\operatorname{std}(r) + \epsilon_{\text{std}}}$$

- 除以 std 是为了把不同 prompt、不同难度的 reward 拉到同一尺度，训练更稳。
- std 用**总体标准差**（`unbiased=False`，分母是 $N$ 而非 $N-1$），对齐 DeepSeek /
  trl 的实现。
- 加 $\epsilon_{\text{std}}$ 防止组内 reward 全相同时除零。

### 裁剪代理目标：和 PPO 完全一样

有了优势之后，用重要性采样比率（importance sampling ratio）去更新策略：

$$r_i^{\text{ratio}} = \exp(\text{logratio}_i) = \frac{\pi_{\text{new}}}{\pi_{\text{old}}}$$

$$\mathcal{L} = -\operatorname{mean}_i\Bigl[\min\bigl(r_i^{\text{ratio}} \cdot A_i,\ \text{clip}(r_i^{\text{ratio}}, 1{-}\epsilon, 1{+}\epsilon) \cdot A_i\bigr)\Bigr]$$

`min` 取悲观下界——如果 ratio 偏离太远（策略变化过大），就用 clip 后的保守值来限
制单步更新幅度。

## 参考实现

```python
import torch

def grpo_loss(logratio, rewards, clip_eps=0.2, eps_std=1e-4):
    # 1. 组内优势归一化
    adv = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + eps_std)

    # 2. PPO 裁剪目标
    ratio = torch.exp(logratio)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    loss = -torch.min(unclipped, clipped).mean()
    return loss
```

## 关键点

1. **std 用总体标准差 `unbiased=False`**。PyTorch 默认 `std()` 是无偏（分母
   $N-1$），GRPO 的论文和 trl 用总体（分母 $N$），必须加 `unbiased=False`，否则在
   组很小（如 G=4）时数值差距明显。

2. **裁剪用 `torch.clamp`，取 `min` 而非 `max`**。`min` 保证无论优势正负都取保
   守的那一侧——正优势时防止 ratio 过大地加速、负优势时防止 ratio 过小地加速反
   向，总之限制「贪心地大步更新」。

3. **初始 loss 约为 0**。`logratio = 0` 时 ratio = 1，clip 无效，loss =
   $-\text{mean}(A)$。而组归一化后 $\text{mean}(A) = 0$（减去均值再除 std，期望就
   是 0），所以训练刚开始 loss 接近 0，很合理。

4. **组内 reward 全相同时 loss 约为 0**。std ≈ 0，加 $\epsilon$ 后分母很大，
   advantage 趋近 0，没有可学的相对信号——这正是预期行为（所有回答一样好，不需
   要区分）。

5. **延伸**：GRPO 和 PPO（见 `pytorch.llm.loss.ppo_clip_loss`）裁剪部分完全一致，
   区别只在优势来源：PPO 用价值网络（value network）+ GAE 估计优势，GRPO 用组内
   z-score。GRPO 省掉了 critic，显存和实现都更轻，是 DeepSeek-R1 等推理模型 RL
   训练的主力。和完全不需要采样/奖励模型（reward model）的 DPO（见
   `pytorch.llm.loss.dpo_loss`）形成互补。
