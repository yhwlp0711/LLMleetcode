# 解题思路：KL 惩罚估计器

## 为什么需要估计器？

RL/RLHF 里我们想约束 `KL(π ‖ π_ref)`，但**没有完整分布**——只有采样出来的
token，以及两个模型对这些 token 的 log-prob。所以要从单样本估计 KL。

设 `logr = log π_ref − log π`（对一个采样 token）。注意
$\mathbb{E}_{\pi}[\text{logr}]$ 恰好是 **reverse KL** `−KL(π‖π_ref)` 的相反数，
即 $\mathbb{E}_\pi[-\text{logr}] = KL(\pi\|\pi_{\text{ref}})$。

## 三个估计器

```python
def _logr(logp, logp_ref):
    return logp_ref - logp

def kl_k1(logp, logp_ref):
    return -_logr(logp, logp_ref)                       # = logp - logp_ref

def kl_k2(logp, logp_ref):
    logr = _logr(logp, logp_ref)
    return 0.5 * logr.pow(2)

def kl_k3(logp, logp_ref):
    logr = _logr(logp, logp_ref)
    return torch.exp(logr) - 1.0 - logr
```

## 三者的取舍

| 估计器 | 无偏? | 方差 | 恒正? | 直觉 |
|---|---|---|---|---|
| **k1** = $-\text{logr}$ | ✅ 无偏 | 大 | ❌ 可为负 | 最朴素，`E[-logr] = KL` |
| **k2** = $\tfrac12\text{logr}^2$ | ❌ 有偏 | 小 | ✅ | 二阶近似，形似平方误差 |
| **k3** = $e^{\text{logr}}-1-\text{logr}$ | ✅ 无偏 | 小 | ✅ | 兼顾无偏与低方差，主流默认 |

### 为什么 k3 无偏又恒正？

- **恒正**：令 $f(x)=e^x-1-x$，$f(0)=0$ 且 $f'(x)=e^x-1$ 在 $x=0$ 变号，
  $x=0$ 是最小值 → $f(x)\ge 0$ 恒成立。
- **无偏**：$\mathbb{E}_\pi[e^{\text{logr}}] = \mathbb{E}_\pi[\pi_{\text{ref}}/\pi] = \sum \pi \cdot \pi_{\text{ref}}/\pi = 1$，
  所以 $\mathbb{E}[e^{\text{logr}}-1]=0$，于是 $\mathbb{E}[k3]=\mathbb{E}[-\text{logr}]=KL$，
  和 k1 同期望但方差更小。

这是 k3 的精妙之处：给 k1 加了一个**期望为 0** 的控制变量 $(e^{\text{logr}}-1)$，
既不改变期望（仍无偏），又抵消了大部分方差。

## 在 RLHF PPO 里怎么用

逐 token 算 KL 惩罚，从 reward 里减掉：

$$r_t' = r_t - \beta\cdot k3(\log\pi_t, \log\pi_{\text{ref},t})$$

再用 `r_t'` 走 GAE。完整流程见 `pytorch.llm.loss.ppo_clip_loss`。

## 与解析 KL 的区别

`pytorch.nn.kl_divergence` 考的是**有完整分布**时的精确 KL
`Σ p·(log p − log q)`。这里的 k1/k2/k3 是**只有采样样本**时对同一个量的
蒙特卡洛估计。两者是「精确值」与「单样本估计」的关系。
