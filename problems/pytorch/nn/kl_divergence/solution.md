# 解题思路：KL 散度

## 核心公式

$$D_{\mathrm{KL}}(P \Vert Q) = \sum_c p_c (\log p_c - \log q_c)$$

## 参考实现

```python
def kl_divergence(p_logits, q_logits):
    log_p = p_logits - torch.logsumexp(p_logits, dim=-1, keepdim=True)
    log_q = q_logits - torch.logsumexp(q_logits, dim=-1, keepdim=True)
    p = log_p.exp()
    kl_per_sample = (p * (log_p - log_q)).sum(dim=-1)   # (N,)
    return kl_per_sample.mean()
```

## 关键点

### 1. 用 log 空间避免数值问题

KL 里有 $\log p_c$ 和 $\log q_c$。直接 `p * log(p/q)` 在 `q_c` 很小时会 `log(≈0) = -inf`。
先在 log 空间算好 `log_p`、`log_q`（稳定 log-softmax），再组合成
`p * (log_p - log_q)`，`p = exp(log_p)`。这样只在需要概率权重时才 `exp`。

### 2. `p_c → 0` 时的项自动归零

理论上 $p_c \log p_c \to 0$（当 $p_c \to 0$）。用 `exp(log_p) * (log_p - log_q)`
时，`p_c` 是精确的小正数，乘出来自然趋于 0，不会产生 `0 * inf = nan`
（因为 `log_p`、`log_q` 都是有限值）。

### 3. reduction 约定

本题按 batch **mean**（除以 N）。注意 PyTorch `F.kl_div` 的坑：
- `F.kl_div(input, target)` 里 **input 要传 log-prob**（对应 `log_q`），target 传 prob（对应 `p`）
- 它算的是 `Σ target·(log target − input) = Σ p·(log p − log q)`，正是 forward KL
- `reduction='batchmean'` 才除以 batch size；`'mean'` 会除以 `N*C`（常见错误）

## forward vs reverse KL

- **forward KL** `KL(P‖Q)`：P 是目标。P 有质量的地方 Q 也必须有 → Q 会
  「覆盖」P 的所有模式（mean-seeking）。知识蒸馏用这个。
- **reverse KL** `KL(Q‖P)`：Q 会集中到 P 的某个高峰（mode-seeking）。
  变分推断、部分 RLHF 用这个。

两者不对称：$KL(P‖Q) \neq KL(Q‖P)$。本题固定考 forward。

## 应用场景

- **知识蒸馏**：student 的 KL 逼近 teacher 的软标签分布
- **RLHF / PPO**：在 reward 里加 `-β·KL(π‖π_ref)` 惩罚，防止 policy 跑偏
- **CE 与 KL**：对固定的真实分布 P，`CE(P,Q) = H(P) + KL(P‖Q)`，
  最小化 CE 等价于最小化 KL

## 解析 KL vs 蒙特卡洛估计

本题算的是**有完整分布**时的精确 KL（`Σ p·(log p−log q)`）。但在 RL/RLHF 里
分布是采样出来的 token，拿不到完整分布，只有 `logπ`、`logπ_ref` 两个标量，
此时用**单样本估计器** k1/k2/k3（见 `pytorch.llm.kl_penalty_estimators`）。
两者是「精确值」与「MC 估计」的关系。
