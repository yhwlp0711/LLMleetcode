# 解题思路：交叉熵损失

## 一句话思路

分类任务最核心的损失。本质就一句话：**取真实类别那一格的 log 概率，取负，再平均**。
两个难点是数值稳定（numerical stability）的 log-softmax（别先 softmax 再 log），以
及正确处理 `ignore_index`（某些样本不计入 loss）。

## 从直觉到公式

### 交叉熵在算什么

对第 $i$ 个样本（真实类别 $y_i$），损失是「模型给正确答案的概率」取负对数：

$$\ell_i = -\log\frac{e^{z_{i,y_i}}}{\sum_c e^{z_{i,c}}} = -\Bigl(z_{i,y_i} - \log\sum_c e^{z_{i,c}}\Bigr)$$

模型越确信正确类别，概率越接近 1，`-log` 越接近 0；越不确信惩罚越大。最终 loss 是所
有**有效样本**的 $\ell_i$ 的平均。

### 数值稳定的 log-softmax

`log(softmax(z))` 里 `exp` 可能溢出、`log(0)` 又变 `-inf`。用 log-sum-exp 恒等式可以
稳定地一步算出 log 概率：

$$\log\text{-softmax}(z)_c = z_c - \log\sum_k e^{z_k}$$

`torch.logsumexp` 内部会先减去 max 再做 `exp`，天然稳定，直接拿来减就行。

### 取真实类别 + 处理 ignore

拿到每行的 log 概率后，用 `gather` 从第 `i` 行取出第 `target[i]` 列，就是 $\ell_i$
需要的那一格。被忽略的样本（`target == ignore_index`，通常是 `-100`）是越界索引，直
接 gather 会报错——技巧是先把它们临时改成合法索引 0 骗过 gather，再用掩码把它们剔
除，最后只对有效样本求均值。

## 参考实现

```python
def cross_entropy(logits, target, ignore_index=-100):
    # 稳定 log-softmax: z - logsumexp(z)
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)  # (N, C)

    valid = target != ignore_index
    safe_target = target.clone()
    safe_target[~valid] = 0                       # 临时改成合法索引，避免 gather 越界
    picked = log_probs.gather(1, safe_target.unsqueeze(1)).squeeze(1)   # 取真实类别

    nll = -picked
    return nll[valid].mean()                       # 只对有效样本平均
```

## 关键点

1. **别先 softmax 再 log**：那样 `exp` 会溢出、`log(0)` 会变 `-inf`。用
   `logits - logsumexp(logits)` 一步算 log 概率，`logsumexp` 内部已经减 max，稳定又简
   洁。这和 sigmoid/softmax 的稳定技巧同源（见 `pytorch.nn.numeric_activations`）。

2. **`gather` 取真实类别的 log 概率**：`log_probs.gather(1, target[:,None])` 等价于
   `log_probs[arange(N), target]`，即每行取出正确类别那一格。

3. **`ignore_index` 的处理套路**：先把越界的 target 临时替换成 0 让 gather 不报错，
   再用 `valid` 掩码剔除，最后**只对有效样本平均**。分母是有效样本数，不是 N——否则有
   被忽略的样本时 loss 会偏小。

4. **延伸**：对 one-hot 标签，交叉熵等于 KL 散度（因为 one-hot 分布的熵为 0），所以
   最小化 CE 等价于最小化 KL——KL 散度的实现见 `pytorch.nn.kl_divergence`。
