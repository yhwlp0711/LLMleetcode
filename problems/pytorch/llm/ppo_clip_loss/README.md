# PPO 损失（GAE + Clipped Surrogate）

PPO 的两个核心：
1. **GAE**（Generalized Advantage Estimation）从 rewards / values 估计优势 A；
2. **裁剪代理损失** 用重要性采样比率约束每步更新幅度。

本题实现「从原始信号到 policy loss」的**完整前向**（不含训练循环 / 反向 / value loss / entropy）。

## 待实现函数

对**单条轨迹**（长度 T）：

```python
def ppo_clip_loss(
    logratio: torch.Tensor,   # (T,)   = log π_new - log π_old（每个时间步）
    rewards: torch.Tensor,    # (T,)   每步即时奖励
    values: torch.Tensor,     # (T+1,) 状态价值 V(s_0..s_T)，最后一个是 bootstrap value
    dones: torch.Tensor,      # (T,)   float，1.0 表示该步是 episode 最后一步（终止）
    gamma: float = 0.99,      # 折扣因子
    lam: float = 0.95,        # GAE 的 λ
    clip_eps: float = 0.2,
) -> torch.Tensor:            # 标量：轨迹平均 loss
```

### 步骤 1：GAE 计算优势

逐步 TD 误差（`dones[t]=1` 时切断下一状态的 bootstrap）：

$$\delta_t = r_t + \gamma\,V(s_{t+1})\,(1 - \text{done}_t) - V(s_t)$$

GAE 优势（从后往前递推）：

$$A_t = \delta_t + \gamma\lambda\,(1 - \text{done}_t)\,A_{t+1}, \qquad A_T \text{ 之后视为 } 0$$

### 步骤 2：裁剪代理损失

$$r_t = \exp(\text{logratio}_t), \qquad
\mathcal{L} = -\operatorname{mean}_t\Bigl[\min\bigl(r_t A_t,\ \text{clip}(r_t, 1-\epsilon, 1+\epsilon)\,A_t\bigr)\Bigr]$$

## 说明

- `logratio` / `rewards` / `dones` 是 `(T,)`；`values` 是 `(T+1,)`（含 bootstrap），都是 `torch.float32`。
- GAE 必须**从后往前**递推；`dones[t]=1` 同时切断 δ 里的 bootstrap 和优势递推项。
- **advantage 不做归一化**（保持确定性）；A 在计算图里视为 detach（本题不涉及反向，直接算数值即可）。
- 裁剪用 `torch.clamp(r, 1-eps, 1+eps)`，objective 取负得到 loss，reduction 为 **mean**。
- 容差 `atol=1e-6`。

> 只考裁剪部分（advantage 直接给）的简化版思路见 `pytorch.llm.grpo_loss`
> （那里 advantage 来自组内 z-score，是另一种估计方式）。
