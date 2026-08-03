# KL 散度（KL Divergence from logits）

KL 散度衡量两个分布的差异，是**知识蒸馏**（student 逼近 teacher）和
**RLHF**（policy 不要偏离 reference 太远）的核心工具。

本题**从 logits 直接算**（内部各自 softmax / log-softmax，数值稳定）。

## 待实现函数

```python
def kl_divergence(
    p_logits: torch.Tensor,   # (N, C) 分布 P 的 logits
    q_logits: torch.Tensor,   # (N, C) 分布 Q 的 logits
) -> torch.Tensor:            # 标量：batch 内逐样本 KL 的平均
```

### 定义（forward KL，即 `KL(P ‖ Q)`）

$$D_{\mathrm{KL}}(P \Vert Q) = \sum_{c} p_c \bigl(\log p_c - \log q_c\bigr)$$

其中 $p = \text{softmax}(p\_logits)$，$q = \text{softmax}(q\_logits)$。

对每个样本各算一个 KL，最终返回**batch 平均**（对 N 求均值）。

## 说明

- 输入都是 `torch.float32` 的 logits（未归一化），形状 `(N, C)`。
- **禁止用** `F.kl_div` / `F.log_softmax` / `F.softmax`，自己实现。
- 用数值稳定的 log-softmax（`z - logsumexp(z)`）；
  KL 内部写成 $\sum_c p_c (\log p_c - \log q_c)$，其中 $p_c$ 用 `exp(log p_c)`。
- 约定 forward KL：`KL(P ‖ Q)`，P 是「目标/teacher」，Q 是「近似/student」。
- reduction 固定为 batch **mean**（除以 N）。
- 容差 `atol=1e-6`。
