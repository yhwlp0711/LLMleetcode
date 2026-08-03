# 解题思路：交叉熵损失

## 核心公式

CE = 负的「真实类别 log-probability」的平均：

$$\ell_i = -\log\text{-softmax}(z_i)_{y_i}, \qquad \mathcal{L} = \frac{1}{|\text{valid}|}\sum_{i \in \text{valid}} \ell_i$$

## 参考实现

```python
def cross_entropy(logits, target, ignore_index=-100):
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)  # (N, C)

    valid = target != ignore_index
    safe_target = target.clone()
    safe_target[~valid] = 0                                    # 避免 gather 越界
    picked = log_probs.gather(1, safe_target.unsqueeze(1)).squeeze(1)

    nll = -picked
    return nll[valid].mean()
```

## 关键点

### 1. 数值稳定的 log-softmax

**不要** `torch.log(torch.softmax(z))`——softmax 里 `exp` 可能上溢/下溢，
`log(0)` 又变 `-inf`。正确做法用 log-sum-exp 恒等式：

$$\log\text{-softmax}(z)_c = z_c - \log\sum_k e^{z_k}$$

`torch.logsumexp` 内部已经先减 max 再 exp，天然稳定。所以一行搞定：

```python
log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
```

### 2. `gather` 取真实类别的 log-prob

`log_probs.gather(1, target[:,None])` 从每行取出第 `target[i]` 列，等价于
`log_probs[torch.arange(N), target]`。

### 3. `ignore_index` 的处理

被忽略的样本 target 是 `-100`（越界），直接 gather 会报错。技巧：先把它们
临时改成合法索引（如 0）以通过 gather，再用 `valid` mask 剔除，最后只对有效
样本求均值。

### 4. 为什么 mean 是「对有效样本」平均

PyTorch 的 `F.cross_entropy(reduction='mean')` 分母是有效样本数，不是 N。
如果拿 N 当分母，有 ignore 时结果会偏小。

## 面试延伸

- **CE 与 NLL 的关系**：`CE(logits) = NLL(log_softmax(logits))`。`F.nll_loss`
  接受的是 log-prob，`F.cross_entropy` 接受 raw logits（内部自己 log_softmax）。
- **CE 与 KL 的关系**：对 one-hot 标签，`CE = H(p) + KL(p‖q) = KL(p‖q)`
  （因为 one-hot 的熵 H(p)=0）。所以最小化 CE 等价于最小化 KL 散度。
