# 解题思路：BCE with logits

## 核心公式

$$\ell = -\bigl[y\log\sigma(z) + (1-y)\log(1-\sigma(z))\bigr]$$

## 数值稳定推导

朴素写法 `-[y*log(σ(z)) + (1-y)*log(1-σ(z))]` 在 `z` 极端时会 `log(0)=-inf`。
展开并合并：

$$
\ell = -y\log\sigma(z) - (1-y)\log(1-\sigma(z))
     = \max(z,0) - z\,y + \log(1 + e^{-|z|})
$$

- `max(z, 0)` 处理 `z` 的正部分，避免 `exp(z)` 上溢
- `log(1 + e^{-|z|})` 里指数参数恒 ≤ 0，不会溢出（用 `torch.log1p` 更精确）

## 参考实现

```python
def bce_with_logits(logits, target):
    z = logits
    loss = z.clamp(min=0) - z * target + torch.log1p(torch.exp(-z.abs()))
    return loss.mean()
```

## 关键点

### `log1p` 而不是 `log(1 + x)`

`torch.log1p(x) = log(1 + x)`，在 `x` 很小时精度更高（避免 `1 + 极小值` 的
浮点吃精度）。

### BCE vs CE

- **CE**（多分类）：一个样本一个类别，softmax 归一化，类别互斥。
- **BCE**（二分类 / 多标签）：每个输出独立 sigmoid，标签可以是多个 1
  （多标签），或单个 0/1（二分类）。
- 二者在「二分类」这个特例下可以互相推导，但实现上 CE 用 log-softmax + gather，
  BCE 用 logsigmoid 逐元素。

### 与 logistic regression 的联系

逻辑回归的损失就是 BCE。题库里 `numpy.ml.logistic_regression` 用的
稳定 sigmoid + BCE，和这里是同一个数值技巧。
