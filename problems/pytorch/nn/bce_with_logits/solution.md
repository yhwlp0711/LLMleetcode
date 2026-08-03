# 解题思路：BCE with logits

## 一句话思路

二分类交叉熵（Binary Cross-Entropy, BCE）的公式很短，难点全在**数值稳定
（numerical stability）**：直接照 `-[y·log σ(z) + (1-y)·log(1-σ(z))]` 写，遇到很大
或很小的 logit 时 `log(0) = -inf` 会炸。核心是把它改写成一个不会溢出的等价式子。

## 从直觉到公式

### 原始定义

对每个元素（logit $z$，标签 $y \in \{0,1\}$）：

$$\ell = -\bigl[y\log\sigma(z) + (1-y)\log(1-\sigma(z))\bigr]$$

$\sigma(z)$ 是模型预测「这是正类」的概率。`y=1` 时惩罚 `-log σ(z)`（预测越接近 1 惩
罚越小），`y=0` 时惩罚 `-log(1-σ(z))`。

### 为什么不能照抄

`z` 很负时 $\sigma(z) \to 0$，`log(0) = -inf`；`z` 很正时 $1-\sigma(z) \to 0$，同样
爆掉。测试会用 `±100` 这种极端 logit 专门戳这个坑。

### 稳定改写

把定义里的 $\sigma$ 展开、合并同类项，可以推出一个数学等价、但永不溢出的形式：

$$\ell = \max(z, 0) - z\cdot y + \log\bigl(1 + e^{-|z|}\bigr)$$

关键在最后一项：指数是 $-|z| \le 0$，`exp` 永远落在 (0, 1]，绝不溢出（overflow）。
`max(z, 0)` 则稳妥地处理了 `z` 的正部分。这正是 PyTorch
`binary_cross_entropy_with_logits` 内部用的形式。

## 参考实现

```python
def bce_with_logits(logits, target):
    z = logits
    # 稳定式: max(z,0) - z*y + log(1 + exp(-|z|))
    loss = z.clamp(min=0) - z * target + torch.log1p(torch.exp(-z.abs()))
    return loss.mean()      # reduction=mean，对所有元素平均
```

## 关键点

1. **核心是让指数 ≤ 0**：用 `-|z|` 当指数，无论 `z` 多大多小，`exp(-|z|)` 都安全。
   这和 softmax「减去 max」是同一类数值稳定思想（见
   `pytorch.nn.numeric_activations`）。

2. **`log1p` 比 `log(1 + x)` 更准**：`torch.log1p(x)` 专门算 `log(1+x)`，在 `x` 很
   小时避免「1 + 极小值」这一步吃掉浮点精度，结果更精确。

3. **`clamp(min=0)` 是 `max(z, 0)` 的写法**：它稳妥地取 `z` 的正部分，是稳定式子里
   不可少的一项，别漏。

4. **BCE 和 CE 的区别**：BCE 是每个输出独立过 sigmoid，适合二分类或多标签（一个样本
   可以有多个 1）；交叉熵（cross-entropy）是一整行过 softmax、类别互斥，适合单标签多
   分类，见 `pytorch.nn.cross_entropy`。二者在二分类特例下可互相推导，但实现不同：BCE
   逐元素，CE 用 log-softmax + gather。

5. **延伸**：逻辑回归（logistic regression）的损失就是 BCE，题库里
   `numpy.ml.logistic_regression` 用的稳定 sigmoid + BCE 和这里是同一套数值技巧。
