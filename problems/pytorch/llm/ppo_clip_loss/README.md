# PPO 裁剪损失（Clipped Surrogate）

PPO 的核心：用「裁剪的重要性采样比率」约束每次更新幅度，防止 policy 一步跑太远。
本题只实现 **clipped surrogate policy loss**（不含 value loss / entropy bonus /
GAE，那些是训练循环的事）。

## 待实现函数

```python
def ppo_clip_loss(
    logratio: torch.Tensor,     # (B,) = log π_new - log π_old（逐样本）
    advantages: torch.Tensor,   # (B,) 优势估计 A
    clip_eps: float = 0.2,
) -> torch.Tensor:              # 标量：batch 平均 loss
```

### 公式

先由 log-ratio 得到概率比率：

$$r = \exp(\log\pi_{\text{new}} - \log\pi_{\text{old}})$$

裁剪目标（PPO 最大化 objective，loss 取负）：

$$\mathcal{L} = -\,\mathbb{E}\Bigl[\min\bigl(r\cdot A,\ \text{clip}(r,\,1-\epsilon,\,1+\epsilon)\cdot A\bigr)\Bigr]$$

其中 $\text{clip}(r, 1-\epsilon, 1+\epsilon)$ 把 $r$ 限制到 $[1-\epsilon, 1+\epsilon]$。
最终对 batch 取**平均**。

## 说明

- `logratio` 与 `advantages` 都是 shape `(B,)` 的 `torch.float32`。
- 用 `torch.clamp(r, 1-eps, 1+eps)` 做裁剪；`torch.min` 取逐元素最小。
- 注意是 **objective 取负** 得到 loss（我们做的是梯度下降最小化 loss）。
- 不对 advantages 做归一化（若需要，调用方在外部处理）。
- reduction 固定为 batch **mean**。
- 容差 `atol=1e-6`。
