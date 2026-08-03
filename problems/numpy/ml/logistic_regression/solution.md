# 解题思路：手撕逻辑回归

## 一句话思路

逻辑回归（Logistic Regression）在线性回归的基础上加了 sigmoid 激活，把输出
压到 (0, 1) 当概率，再用二分类交叉熵（Binary Cross-Entropy）作为损失。难点
有两个：**sigmoid 的数值稳定（numerical stability）**和**梯度的正确推导**。

## 从直觉到公式

### 模型

先做线性变换 $z = Xw + b$，再用 sigmoid 把它映射到概率：

$$p = \sigma(z) = \frac{1}{1 + e^{-z}}$$

$p$ 表示「样本属于类别 1 的概率」。

### 损失：二分类交叉熵

$$L = -\frac{1}{N}\sum_{i=1}^{N}\bigl[y_i \log p_i + (1 - y_i)\log(1 - p_i)\bigr]$$

直觉：当 $y=1$ 时，我们希望 $p$ 接近 1，这样 $\log p$ 接近 0 → loss 小；
反之亦然。

### 梯度化简后非常优美

把 $\partial L / \partial z_i$ 展开化简后，会得到一个极为简洁的结果：

$$\frac{\partial L}{\partial z_i} = p_i - y_i$$

继续链式法则求 $w$ 和 $b$：

$$\nabla_w = \frac{1}{N} X^\top (p - y), \quad \nabla_b = \frac{1}{N}\sum_i(p_i - y_i)$$

注意系数是 $1/N$（不是 $2/N$），因为交叉熵公式里没有平方。

### sigmoid 为什么要分支处理？

朴素写法 `1/(1+exp(-z))`，当 $z$ 很负时 `exp(-z)` 巨大 → 溢出（overflow）。
解决办法：**让 `exp` 的指数永远 ≤ 0**：

- $z \ge 0$：用 $1/(1+e^{-z})$，指数 $-z \le 0$ ✅
- $z < 0$：用等价形式 $e^z/(1+e^z)$，指数 $z < 0$ ✅

两式分子分母同乘 $e^z$ 即可互换，数学上完全相等。

## 参考实现

```python
import numpy as np

def _sigmoid(z):
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out

def fit_predict_proba(X_train, y_train, X_test, *, lr, epochs):
    N, D = X_train.shape
    w = np.zeros(D, dtype=np.float64)
    b = 0.0

    for _ in range(epochs):
        z = X_train @ w + b
        p = _sigmoid(z)
        err = p - y_train                               # 梯度的核心：p - y
        w -= lr * (X_train.T @ err) / N
        b -= lr * err.sum() / N

    proba_test = _sigmoid(X_test @ w + b)
    return w, float(b), proba_test
```

## 关键点

1. **sigmoid 分支保证数值安全**：不做分支处理，`z = -1000` 时 `exp(1000)`
   直接 `inf`，再传播到梯度全变 NaN，整轮训练崩掉。这个技巧和
   `numpy.ml.linear_regression` 里的线性回归相比是最大的额外难点。

2. **梯度系数是 `1/N` 不是 `2/N`**：交叉熵公式里没有平方那个 $\frac{1}{2}$
   对消项。照搬线性回归的 `2/N` 会导致有效学习率翻倍，数值不一致。

3. **`p - y` 就是整个反向传播的入口**：这个漂亮的化简省去了分别对 $\log p$
   和 $\log(1-p)$ 求导的麻烦——一步到位。这也是 sigmoid + cross-entropy
   组合在理论上的优雅之处。

4. **返回的是概率，不是标签**：`proba_test = sigmoid(X_test @ w + b)`，
   值域 (0, 1)。如果要 0/1 标签再手动 `(proba > 0.5).astype(int64)`，但
   本题只要概率。

5. **延伸**：这里的稳定 sigmoid 技巧和 `pytorch.nn.numeric_activations` 里
   手写 sigmoid 完全一样。梯度下降框架则和 `numpy.ml.linear_regression`
   共享——只是激活和损失不同。
