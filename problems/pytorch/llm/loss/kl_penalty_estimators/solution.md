# 解题思路：KL 惩罚估计器

## 一句话思路

RLHF / RL 训练中想约束策略不偏离参考模型太远，需要逐 token 估计
$\text{KL}(\pi \| \pi_{\text{ref}})$。但手上没有完整分布——只有两个模型对**同一
批 token** 的 log 概率。三个经典估计器（k1 / k2 / k3）就是从这对 log-prob 做单
样本蒙特卡洛估计。核心就是一个中间量 `logr = logp_ref - logp`。

## 从直觉到公式

### 为什么需要估计器？

精确 KL 需要遍历整个词表算 $\sum_x \pi(x)\log\frac{\pi(x)}{\pi_{\text{ref}}(x)}$，
计算量大。RLHF 训练时已经从 $\pi$ 采样了 token，手上只有 `logp`（$\log\pi$）和
`logp_ref`（$\log\pi_{\text{ref}}$）这两个标量。我们想要的是
$\mathbb{E}_\pi[-\text{logr}] = \text{KL}(\pi \| \pi_{\text{ref}})$ 的单样本估计。

定义：

$$\text{logr} = \log\pi_{\text{ref}} - \log\pi = \log\frac{\pi_{\text{ref}}}{\pi}$$

### 三个估计器

| 估计器 | 公式 | 无偏？ | 恒 ≥ 0？ | 方差 |
|---|---|---|---|---|
| **k1** | $-\text{logr}$ | 是 | 否（可为负） | 大 |
| **k2** | $\frac{1}{2}\,\text{logr}^2$ | 否（有偏） | 是 | 小 |
| **k3** | $e^{\text{logr}} - 1 - \text{logr}$ | 是 | 是 | 小 |

k1 是最朴素的：期望就是 KL，但方差大、可以取负值。k2 像平方误差，恒正方差小但有
偏。k3 兼顾了无偏、低方差和恒正——是 trl / DeepSeek 等主流实现的默认。

### k3 为什么既无偏又恒正？

- **恒正**：令 $f(x) = e^x - 1 - x$，$f(0) = 0$，$f'(x) = e^x - 1$ 在 $x=0$
  处过零且只此一处极值 → $f(x) \ge 0$ 恒成立。
- **无偏**：$\mathbb{E}_\pi[e^{\text{logr}}] = \mathbb{E}_\pi[\pi_{\text{ref}}/\pi] = \sum \pi \cdot \pi_{\text{ref}}/\pi = 1$，
  所以 $\mathbb{E}[e^{\text{logr}} - 1] = 0$，于是
  $\mathbb{E}[\text{k3}] = \mathbb{E}[-\text{logr}] = \text{KL}$。

k3 的巧思：在 k1（$-\text{logr}$）上加了一个**期望为 0** 的项
$(e^{\text{logr}} - 1)$，不影响无偏性，但大幅压低方差（控制变量技巧）。

## 参考实现

```python
import torch

def _logr(logp, logp_ref):
    return logp_ref - logp

def kl_k1(logp, logp_ref):
    return -_logr(logp, logp_ref)              # = logp - logp_ref

def kl_k2(logp, logp_ref):
    logr = _logr(logp, logp_ref)
    return 0.5 * logr.pow(2)

def kl_k3(logp, logp_ref):
    logr = _logr(logp, logp_ref)
    return torch.exp(logr) - 1.0 - logr
```

## 关键点

1. **`logr` 的方向：分子是 ref，分母是 policy**。`logr = logp_ref - logp`。三个
   估计器都用同一个 logr，方向搞反会估计成反向 KL。

2. **返回逐元素估计，不做 reduction**。三个函数输入输出同 shape，不要 mean/sum，
   让调用方自己决定怎么聚合。

3. **k3 在 RLHF PPO 中当 KL 惩罚**。逐 token 算完 k3 后乘以系数 $\beta$、从
   reward 里减掉，再走 GAE。完整流程见 `pytorch.llm.loss.ppo_clip_loss`。

4. **k1 可以为负，作为惩罚项时可能「奖励」偏离 ref**。这是 k1 方差大的表现——某
   些 token 上 policy 比 ref 概率更高时 k1 < 0。k3 恒正就没这个问题。

5. **延伸**：这里的 k1/k2/k3 是**只有采样样本时**的蒙特卡洛估计；如果拿到完整
   分布，直接精确求和即可（那是另一种场景）。GRPO（见
   `pytorch.llm.loss.grpo_loss`）在一些实现里也用 k3 做 per-token KL 惩罚。
