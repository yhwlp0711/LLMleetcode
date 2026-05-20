# 解题思路：手撕逻辑回归（NumPy）

## 1. 关键数值技巧：稳定的 sigmoid

朴素写法 `1 / (1 + np.exp(-z))` 在 `z = -1000` 时 `exp(1000)` 溢出，得到
`inf` 然后变成 0；在 `z = 1000` 时 `exp(-1000)` 下溢为 0，结果是 1，看起
来好像没坏，但梯度会消失。

**等价但稳定的写法**：让 `exp` 的参数永远 ≤ 0。

```python
def _sigmoid(z):
    out = np.empty_like(z)
    pos = z >= 0
    out[pos]  = 1.0 / (1.0 + np.exp(-z[pos]))     # 对正 z，exp(-z) ∈ (0, 1]
    e = np.exp(z[~pos])                            # 对负 z，exp(z) ∈ (0, 1]
    out[~pos] = e / (1.0 + e)
    return out
```

数学上 `e^z / (1 + e^z) = 1 / (1 + e^{-z})`，所以两种写法等价。**用** `where`
**也可以**：

```python
def _sigmoid(z):
    abs_z = np.abs(z)
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-abs_z)),
                            np.exp(-abs_z) / (1.0 + np.exp(-abs_z)))
```

## 2. 梯度推导

二分类交叉熵：
$L = -\frac{1}{N}\sum_i [y_i \log p_i + (1-y_i)\log(1-p_i)]$，其中
$p_i = \sigma(z_i)$，$z_i = w^\top x_i + b$。

`dL/dp_i` 与 `dp_i/dz_i = p_i(1-p_i)` 凑一起，得到非常优美的形式：

$$\frac{\partial L}{\partial z_i} = p_i - y_i$$

继续链式求 `w` 和 `b`：

$$\nabla_w = \frac{1}{N} X^\top (p - y),\quad \nabla_b = \frac{1}{N}\sum (p - y)$$

**注意**：这里是 `1/N`，不像线性回归是 `2/N`（因为损失里没有平方系数）。

## 3. 参考实现

```python
def fit_predict_proba(X_train, y_train, X_test, *, lr, epochs):
    N, D = X_train.shape
    w = np.zeros(D, dtype=np.float64)
    b = 0.0

    for _ in range(epochs):
        z = X_train @ w + b
        p = _sigmoid(z)
        err = p - y_train
        w -= lr * (X_train.T @ err) / N
        b -= lr * err.sum() / N

    proba_test = _sigmoid(X_test @ w + b)
    return w, float(b), proba_test
```

## 4. 易错点

1. **sigmoid 不稳定** → `nan` 直接传染到梯度，整轮训练崩溃。
2. **梯度系数搞错**：交叉熵的 `1/N`，不要照搬线性回归的 `2/N`。
3. **直接对 `log(p)` 求导而不简化**：会做大量重复计算，还容易在 `p` 接近 0
   或 1 时数值不稳定（`log(0) = -inf`）。手算化简到 `p - y` 再实现。
4. **预测时返回概率还是标签**：本题要求返回概率，所以是 `proba_test`，不
   是 `argmax` 后的 0/1。
