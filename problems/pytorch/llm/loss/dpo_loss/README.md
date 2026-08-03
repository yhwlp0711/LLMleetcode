# DPO 损失（Direct Preference Optimization）

DPO 是 RLHF 的免 reward-model / 免 PPO 替代方案：直接用「偏好对」
（chosen 优于 rejected）训练 policy。本题只实现**loss 的前向计算**
（不涉及训练循环、不涉及如何得到 log-prob）。

## 待实现函数

输入是已经算好的 **序列级 log-probability**（对整段回答的 log π 求和）：

```python
def dpo_loss(
    policy_chosen_logps: torch.Tensor,    # (B,) policy 对 chosen 的 log-prob
    policy_rejected_logps: torch.Tensor,  # (B,) policy 对 rejected 的 log-prob
    ref_chosen_logps: torch.Tensor,       # (B,) reference 对 chosen 的 log-prob
    ref_rejected_logps: torch.Tensor,     # (B,) reference 对 rejected 的 log-prob
    beta: float = 0.1,
) -> torch.Tensor:                        # 标量：batch 平均 loss
```

### 公式

先算 policy 相对 reference 的 log-ratio（chosen 与 rejected 各一个）：

$$
\begin{aligned}
\Delta_{\text{chosen}} &= \log\pi_\theta(y_w\mid x) - \log\pi_{\text{ref}}(y_w\mid x) \\
\Delta_{\text{rejected}} &= \log\pi_\theta(y_l\mid x) - \log\pi_{\text{ref}}(y_l\mid x)
\end{aligned}
$$

DPO loss（Bradley-Terry 偏好模型下）：

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\Bigl[\log\sigma\bigl(\beta\,(\Delta_{\text{chosen}} - \Delta_{\text{rejected}})\bigr)\Bigr]$$

其中 $\sigma$ 是 sigmoid。最终对 batch 取**平均**。

## 说明

- 四个输入都是 shape `(B,)` 的 `torch.float32`（序列级 log-prob，已 detach 的 ref 不用你管）。
- **要求数值稳定**：用 `logsigmoid` 而不是 `log(sigmoid(...))`（后者在负值区会下溢）。
  可以用 `torch.nn.functional.logsigmoid`，或自己实现 `logsigmoid(x) = -softplus(-x)`。
- reduction 固定为 batch **mean**。
- 容差 `atol=1e-6`。

> 提示：`log σ(x)` 与 `softplus` 的关系是 `log σ(x) = -log(1+e^{-x}) = -softplus(-x)`。
