# KL 惩罚估计器（k1 / k2 / k3）

RLHF / RL 里通常**无法解析计算** KL——分布是采样出来的 token，手上只有
当前 policy 和参考模型对**同一批 token** 的 log-probability。于是用蒙特卡洛
估计器从单样本估计 `KL(π ‖ π_ref)`。这是 John Schulman 的经典三估计器
（[Approximating KL Divergence](http://joschu.net/blog/kl-approx.html)）。

## 记号

对每个 token，令

$$\text{logr} = \log\pi_{\text{ref}} - \log\pi = \log\frac{\pi_{\text{ref}}}{\pi}$$

（注意方向：分子是 ref，分母是当前 policy。）

## 待实现函数

三个函数，输入都是同 shape 的 `logp`（当前 π 的 log-prob）和 `logp_ref`
（参考模型的 log-prob），返回**逐元素**的 KL 估计（同 shape，不做 reduction）：

```python
def kl_k1(logp: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    # k1 = -logr = logp - logp_ref
    ...

def kl_k2(logp: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    # k2 = 0.5 * logr**2
    ...

def kl_k3(logp: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    # k3 = (exp(logr) - 1) - logr
    ...
```

### 三个估计器

| 估计器 | 公式 | 性质 |
|---|---|---|
| **k1** | $-\text{logr}$ | 无偏，但方差大，可能取负值 |
| **k2** | $\tfrac12\,\text{logr}^2$ | 有偏，方差小，恒 ≥ 0 |
| **k3** | $(e^{\text{logr}} - 1) - \text{logr}$ | **无偏 + 低方差 + 恒 ≥ 0**（主流实现默认） |

## 说明

- 输入是任意同 shape 的 `torch.float32`，返回同 shape 的逐元素估计（**不要** mean/sum）。
- 三者都用 `logr = logp_ref - logp` 这一方向。
- k3 是 trl / DeepSeek 等实现里 GRPO/PPO 默认用的估计器。
- 容差 `atol=1e-6`。
