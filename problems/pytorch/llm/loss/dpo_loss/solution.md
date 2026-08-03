# 解题思路：DPO 损失

## 一句话思路

DPO（Direct Preference Optimization，直接偏好优化）想让模型「更爱说 chosen
（人类偏好的回答）、少说 rejected（被拒绝的回答）」。它把这件事变成一个
**二分类问题**：给每对回答算一个「chosen 比 rejected 好多少」的分数，再用
sigmoid 把它推向「越大越好」。所以最终 loss 就是一行 `-logsigmoid(β·margin)`。

## 从直觉到公式

### 「好多少」怎么量化？

我们没有一个现成的打分器。DPO 的巧思是：用**当前模型相对初始模型的变化量**
来当隐式的分数。对一条回答 $y$，定义

$$\Delta(y) = \log\pi_\theta(y) - \log\pi_{\text{ref}}(y)$$

其中 $\pi_\theta$ 是正在训练的模型（policy），$\pi_{\text{ref}}$ 是训练前
的初始模型（reference，冻结不动）。$\Delta$ 大，说明「训练后模型比原来更愿意
生成这条回答」。

### 组装成 loss

我们希望 chosen 的 $\Delta$ 比 rejected 的大，即 margin（差距）为正：

$$\text{margin} = \Delta(y_w) - \Delta(y_l) \quad (\text{w=chosen, l=rejected})$$

套上 Bradley-Terry 偏好模型（把「A 比 B 好」建模成 $\sigma(\text{分差})$），
最大化偏好概率等价于最小化：

$$\mathcal{L}_{\text{DPO}} = -\log\sigma\bigl(\beta \cdot \text{margin}\bigr)$$

margin 越大 → $\sigma \to 1$ → loss → 0；搞反了 → loss 快速变大。

## 参考实现

```python
import torch.nn.functional as F

def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    delta_chosen   = policy_chosen_logps   - ref_chosen_logps    # Δ(y_w)
    delta_rejected = policy_rejected_logps - ref_rejected_logps  # Δ(y_l)
    logits = beta * (delta_chosen - delta_rejected)              # β·margin
    loss = -F.logsigmoid(logits)
    return loss.mean()
```

## 关键点

1. **为什么要减去 reference？** 一是 $\Delta = \log(\pi_\theta/\pi_{\text{ref}})$
   衡量「相对初始模型改变了多少」，DPO 的理论推导正好表明最优策略满足
   $\log(\pi_\theta/\pi_{\text{ref}}) \propto$ 奖励；二是它起到**正则化
   （regularization）**作用，拉住模型别偏离初始模型太远、避免训崩。

2. **用 `logsigmoid`，别写成 `log(sigmoid(x))`**。当 `x` 很负时 `sigmoid(x)`
   趋近 0，`log(0) = -inf` 就爆了。`F.logsigmoid` 内部用等价但稳定的
   $-\log(1+e^{-x})$ 直接算，全程不会溢出（overflow）。这和 sigmoid/softmax
   题里「让 exp 指数 ≤ 0」是同一类数值稳定（numerical stability）技巧。

3. **几个边界帮助理解 loss 的含义**：
   - margin = 0（模型对两条回答一样偏好）：`loss = -log σ(0) = log 2 ≈ 0.69`，
     这是「还没学到任何偏好」的基准值。
   - chosen 远好于 rejected：margin 很大，`loss → 0`。
   - 学反了（更爱 rejected）：margin 为负，loss 迅速增大，惩罚很重。

4. **`beta` 控制什么？** 它是 margin 的放大系数，决定 loss 对偏好差距有多
   敏感：`beta` 越大越「较真」（margin 稍有不对就重罚、也越容易训崩），
   越小越温和。常用 0.1 ~ 0.5。

5. **延伸**：DPO 最大的好处是把「训练奖励模型 + 强化学习」这套复杂流程，
   压缩成一个像分类一样的监督损失——不需要采样、不需要奖励模型（reward
   model）、不需要价值网络（value network）。它和需要这些组件的 PPO
   （见 `pytorch.llm.loss.ppo_clip_loss`）形成鲜明对比。
