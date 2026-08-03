# 解题思路：PPO 损失（KL 惩罚 + GAE + Clipped Surrogate）

## 一句话思路

RLHF 里 PPO 的 policy loss 分三步：先把 **KL 惩罚**逐 token 并入 reward（约束策
略别跑离参考模型太远）；再用带惩罚的 reward + 价值估计走 **GAE**（广义优势估计）
从后往前递推出每步优势（advantage）；最后用**裁剪代理目标**限制每步更新幅度。

## 从直觉到公式

### 步骤 1：KL 惩罚并入 reward

RLHF 想要模型变得更好（reward 大），但又不能变得「面目全非」。做法是逐 token 从
reward 里减掉一个 KL 项，相当于给跑偏的行为「罚钱」。KL 用 k3 估计器（无偏 + 恒
正 + 低方差，见 `pytorch.llm.loss.kl_penalty_estimators`）：

$$\text{logr}_t = \log\pi_{\text{ref},t} - \log\pi_t$$

$$\text{kl}_t = e^{\text{logr}_t} - 1 - \text{logr}_t$$

$$r'_t = r_t - \beta \cdot \text{kl}_t$$

### 步骤 2：GAE 从后往前递推优势

有了带惩罚的 reward $r'$ 和外部给的 value 估计 $V(s_t)$，先算单步 TD 误差
（temporal difference）：

$$\delta_t = r'_t + \gamma \cdot V(s_{t+1}) \cdot (1 - \text{done}_t) - V(s_t)$$

再从后往前递推得到**多步**优势：

$$A_t = \delta_t + \gamma \lambda \cdot (1 - \text{done}_t) \cdot A_{t+1}$$

$A_T$ 之后初始化为 0。$\lambda$ 是偏差-方差的插值旋钮：$\lambda=0$ 只用一步 TD
（低方差高偏差），$\lambda=1$ 相当于 Monte-Carlo（高方差低偏差），常用 0.95。

`(1 - done_t)` 在终止步把 bootstrap 和递推都切断——episode 结束后的回报不应回传。

### 步骤 3：裁剪代理目标

$$\text{ratio}_t = \exp(\text{logratio}_t) = \frac{\pi_{\text{new}}}{\pi_{\text{old}}}$$

$$\mathcal{L} = -\operatorname{mean}_t\Bigl[\min\bigl(\text{ratio}_t \cdot A_t,\ \text{clip}(\text{ratio}_t, 1{-}\epsilon, 1{+}\epsilon) \cdot A_t\bigr)\Bigr]$$

`min` 保证无论优势正负都取保守方向——限制策略单步变化不能太大，避免「一步迈太远
训崩」。

## 参考实现

```python
import torch

def ppo_clip_loss(logratio, logp, logp_ref, rewards, values, dones,
                  gamma=0.99, lam=0.95, clip_eps=0.2, kl_coef=0.1):
    # Step 1: KL penalty (k3) 并入 reward
    logr = logp_ref - logp
    kl = torch.exp(logr) - 1.0 - logr
    r = rewards - kl_coef * kl

    # Step 2: GAE（从后往前递推）
    T = r.shape[0]
    adv = torch.zeros(T, dtype=r.dtype)
    gae = torch.zeros((), dtype=r.dtype)
    for t in range(T - 1, -1, -1):
        nonterminal = 1.0 - dones[t]
        delta = r[t] + gamma * values[t + 1] * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae

    # Step 3: 裁剪代理目标
    ratio = torch.exp(logratio)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    loss = -torch.min(unclipped, clipped).mean()
    return loss
```

## 关键点

1. **KL 惩罚进 reward，不是直接进 loss**。把 $-\beta \cdot \text{kl}$ 加到 reward
   上，让 GAE 把惩罚也「折扣传播」到多步优势里——这是 InstructGPT / trl 的标准
   做法。

2. **`values` 长度是 T+1**。多出来的 `values[T]` 是 bootstrap value（对最终状态
   的价值估计）。`nonterminal = 1 - dones[t]` 在终止步把 `values[t+1]` 乘没，同
   时也切断优势递推项。

3. **必须从后往前递推**。$A_t$ 依赖 $A_{t+1}$，所以 `range(T-1, -1, -1)`。初始
   `gae = 0`，代表 $A_T$ 之后的未来贡献为 0。

4. **advantage 不做归一化**。工业训练里常把 advantage 减 mean 除 std 来稳定梯度，
   但本题为了确定性判分，不做归一化，直接用算出来的原始值。

5. **`min` 在裁剪目标里的作用**。正优势时 `min` 防止 ratio 过大（策略变得过于偏
   好该动作）；负优势时 `min` 防止 ratio 过小（策略「过度逃离」该动作）。无论哪
   种情况，都是取保守的那一侧。

6. **延伸**：PPO 和 GRPO（见 `pytorch.llm.loss.grpo_loss`）裁剪部分完全相同，区
   别在优势来源：PPO 用价值网络（value network）+ GAE，GRPO 用组内 z-score 无
   需 critic。不需要采样的纯监督做法则是 DPO（见 `pytorch.llm.loss.dpo_loss`）。
