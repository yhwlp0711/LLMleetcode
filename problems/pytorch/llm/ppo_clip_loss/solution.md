# 解题思路：PPO 损失（GAE + Clipped Surrogate）

## 两段结构

PPO 的 policy loss = 「GAE 算优势」+「裁剪代理目标」。本题把两段合到一个函数里。

## 步骤 1：GAE

### TD 误差与递推

$$\delta_t = r_t + \gamma V(s_{t+1})(1-\text{done}_t) - V(s_t)$$
$$A_t = \delta_t + \gamma\lambda(1-\text{done}_t)A_{t+1}$$

**从后往前**递推，`A_T` 之后初始化为 0。

```python
T = rewards.shape[0]
adv = torch.zeros(T)
gae = torch.zeros(())
for t in range(T - 1, -1, -1):
    nonterminal = 1.0 - dones[t]
    delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
    gae = delta + gamma * lam * nonterminal * gae
    adv[t] = gae
```

### 关键点

- **`values` 长度 T+1**：多出来的 `values[T]` 是 bootstrap value（对最后状态
  的价值估计）。若最后一步是终止步（`dones[T-1]=1`），`nonterminal=0` 会把它
  乘没，正确切断。
- **`dones[t]` 同时作用两处**：δ 里的 bootstrap 项 和 优势递推项。终止步之后
  的 return 不应跨 episode 传播。
- **必须从后往前**：`A_t` 依赖 `A_{t+1}`。

### λ 的直觉

- `λ=0`：`A_t = δ_t`，只用一步 TD（低方差、高偏差）。
- `λ=1`：`A_t = Σ γ^k δ_{t+k}`，等价 Monte-Carlo 优势（高方差、低偏差）。
- `λ∈(0,1)`：在偏差-方差间插值。典型 0.95。

## 步骤 2：裁剪代理损失

和纯裁剪版一样：

```python
ratio = torch.exp(logratio)
unclipped = ratio * adv
clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
loss = -torch.min(unclipped, clipped).mean()
```

`min` 取悲观下界；objective 取负得到 loss。详见对「clip 为何用 min」的分析
（同 GRPO 里的裁剪部分）。

## 边界检查

- **所有步终止**（`dones` 全 1）：`nonterminal=0`，递推项和 bootstrap 都被切，
  `A_t = r_t - V(s_t)`。此时若 `logratio=0`，`loss = -mean(r - V[:T])`。
- **`logratio=0`**（`ratio=1`，还没更新）：`loss = -mean(A)`。

## 为什么 advantage 通常 detach / 不参与 policy 梯度

在完整训练里，advantage 由 critic 给出，policy loss 对 advantage **不回传梯度**
（advantage 当常数）。本题只算数值，不涉及反向，直接算即可。

## PPO vs GRPO

两者裁剪部分完全相同，区别在优势来源：
- **PPO**：GAE（需要 critic 提供 `values`）。
- **GRPO**（见 `pytorch.llm.grpo_loss`）：组内 reward z-score，**无需 critic**。
