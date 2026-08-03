# PPO 损失（KL 惩罚 + GAE + Clipped Surrogate）

RLHF 语境下 PPO 的完整 policy-loss 前向，三步：
1. **KL 惩罚**：逐 token 把 `−β·KL(π‖π_ref)` 加到 reward 上，约束 policy 别跑离参考模型太远；
2. **GAE**（Generalized Advantage Estimation）从带惩罚的 reward + values 估计优势 A；
3. **裁剪代理损失** 用重要性采样比率约束每步更新幅度。

本题实现「从原始信号到 policy loss」的**完整前向**（不含训练循环 / 反向 / value loss / entropy）。

## 待实现函数

对**单条轨迹**（长度 T）：

```python
def ppo_clip_loss(
    logratio: torch.Tensor,   # (T,)   = log π_new - log π_old（每个时间步）
    logp: torch.Tensor,       # (T,)   当前 policy 对采样 token 的 log-prob
    logp_ref: torch.Tensor,   # (T,)   参考模型对同一 token 的 log-prob
    rewards: torch.Tensor,    # (T,)   每步即时（任务）奖励
    values: torch.Tensor,     # (T+1,) 状态价值 V(s_0..s_T)，最后一个是 bootstrap value
    dones: torch.Tensor,      # (T,)   float，1.0 表示该步是 episode 最后一步（终止）
    gamma: float = 0.99,      # 折扣因子
    lam: float = 0.95,        # GAE 的 λ
    clip_eps: float = 0.2,
    kl_coef: float = 0.1,     # KL 惩罚系数 β
) -> torch.Tensor:            # 标量：轨迹平均 loss
```

### 步骤 1：KL 惩罚并入 reward

用 **k3 估计器**（`logr = logp_ref - logp`）逐 token 估计 KL，从 reward 里减掉：

$$\text{kl}_t = e^{\text{logr}_t} - 1 - \text{logr}_t, \qquad r_t' = r_t - \beta\cdot\text{kl}_t$$

### 步骤 2：GAE 计算优势（用 `r'`）

$$\delta_t = r_t' + \gamma\,V(s_{t+1})\,(1 - \text{done}_t) - V(s_t)$$
$$A_t = \delta_t + \gamma\lambda\,(1 - \text{done}_t)\,A_{t+1}, \qquad A_T \text{ 之后视为 } 0$$

### 步骤 3：裁剪代理损失

$$r_t = \exp(\text{logratio}_t), \qquad
\mathcal{L} = -\operatorname{mean}_t\Bigl[\min\bigl(r_t A_t,\ \text{clip}(r_t, 1-\epsilon, 1+\epsilon)\,A_t\bigr)\Bigr]$$

## 说明

- `logratio` / `logp` / `logp_ref` / `rewards` / `dones` 是 `(T,)`；`values` 是 `(T+1,)`（含 bootstrap），都是 `torch.float32`。
- **KL 惩罚固定用 k3 估计器**（`exp(logr)-1-logr`，`logr = logp_ref - logp`），逐 token 施加到 reward 后再走 GAE。见 `pytorch.llm.loss.kl_penalty_estimators`。
- GAE 必须**从后往前**递推；`dones[t]=1` 同时切断 δ 里的 bootstrap 和优势递推项。
- **advantage 不做归一化**（保持确定性）；本题不涉及反向，直接算数值即可。
- 裁剪用 `torch.clamp(r, 1-eps, 1+eps)`，objective 取负得到 loss，reduction 为 **mean**。
- 容差 `atol=1e-6`。

> 无 critic 版（advantage 来自组内 z-score）见 `pytorch.llm.loss.grpo_loss`。
