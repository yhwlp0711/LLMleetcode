# 解题思路：KL 散度

## 一句话思路

KL 散度（KL divergence）衡量两个概率分布差多远，是知识蒸馏（knowledge
distillation）和 RLHF 的核心工具。本题从 logits 直接算，难点是**全程在 log 空间操
作**保证数值稳定（numerical stability），只在需要概率权重时才 `exp`。

## 从直觉到公式

### KL 在量什么

前向 KL（forward KL）`KL(P‖Q)` 度量「用 Q 去近似 P」的信息损失：

$$D_{\mathrm{KL}}(P \Vert Q) = \sum_c p_c\bigl(\log p_c - \log q_c\bigr)$$

$p_c$ 在某类概率高、而 $q_c$ 却低时，$\log p_c - \log q_c$ 很大，这一项贡献大惩罚；
两个分布完全一致时 KL = 0。这里 $p = \text{softmax}(p\_logits)$，
$q = \text{softmax}(q\_logits)$。

### 为什么要待在 log 空间

如果先算出概率再 `p * log(p/q)`，当 $q_c$ 很小时 `log(≈0) = -inf` 就炸了。稳妥做法
是先用稳定的 log-softmax 算好 `log_p`、`log_q`：

$$\log\text{-softmax}(z) = z - \log\sum_k e^{z_k}$$

（`torch.logsumexp` 内部已减 max，天然稳定。）再组合成 `p * (log_p - log_q)`，其中
概率权重 `p = exp(log_p)`。这样 `log_p`、`log_q` 始终是有限值，$p_c \to 0$ 的项自然
趋于 0，不会出现 `0 * inf = nan`。

## 参考实现

```python
def kl_divergence(p_logits, q_logits):
    log_p = p_logits - torch.logsumexp(p_logits, dim=-1, keepdim=True)  # 稳定 log-softmax
    log_q = q_logits - torch.logsumexp(q_logits, dim=-1, keepdim=True)
    p = log_p.exp()                                    # 只在需要权重时才 exp
    kl_per_sample = (p * (log_p - log_q)).sum(dim=-1)  # 逐样本 KL, (N,)
    return kl_per_sample.mean()                        # batch 平均
```

## 关键点

1. **全程 log 空间是稳定的关键**：直接 `log(p/q)` 在 `q` 接近 0 时会 `-inf`。先算好
   `log_p`、`log_q` 再相减，避免了这个坑。这和交叉熵里用 log-softmax 是同一招（见
   `pytorch.nn.cross_entropy`）。

2. **`p_c → 0` 的项自动归零**：理论上 $p_c \log p_c \to 0$。因为 `p = exp(log_p)` 是
   精确的小正数、`log_p - log_q` 是有限值，乘出来自然趋于 0，不会产生 `nan`。

3. **前向 KL vs 反向 KL**：`KL(P‖Q)`（前向）要求「P 有质量的地方 Q 也得有」，Q 会去
   覆盖 P 的所有峰（mean-seeking），知识蒸馏用它；`KL(Q‖P)`（反向）会让 Q 集中到 P 的
   某个峰（mode-seeking）。两者不对称，本题固定考前向。

4. **reduction 约定**：本题按 batch 平均（除以 N）。用 PyTorch `F.kl_div` 时要小
   心——它的 `input` 要传 log 概率、`target` 传概率，且只有 `reduction='batchmean'`
   才除以 batch size，`'mean'` 会除以 `N*C`。

5. **延伸**：本题算的是**有完整分布**时的精确 KL。但在 RLHF / PPO 里，token 是采样出
   来的，只拿得到 `logπ`、`logπ_ref` 两个标量，此时改用单样本估计器 k1/k2/k3——见
   `pytorch.llm.loss.kl_penalty_estimators`，两者是「精确值」与「蒙特卡洛估计」的关系。
