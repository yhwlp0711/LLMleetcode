# 解题思路：手撕线性回归（NumPy）

## 1. 数学推导

线性回归模型：$\hat{y}\_i = w^\top x\_i + b$，目标最小化 MSE：

$$L = \frac{1}{N}\sum\_i (\hat{y}\_i - y\_i)^2$$

对 $w$ 求梯度（用链式法则，把误差记为 $e\_i = \hat{y}\_i - y\_i$）：

$$\frac{\partial L}{\partial w} = \frac{2}{N}\sum\_i e\_i \cdot x\_i = \frac{2}{N} X^\top e$$

$$\frac{\partial L}{\partial b} = \frac{2}{N}\sum\_i e\_i$$

注意 **常数 2 不能省**，省掉等价于把学习率减半，对绝对数值有影响（判分会
逐位比对）。

## 2. 向量化实现

```python
def fit_predict(X_train, y_train, X_test, *, lr, epochs):
    N, D = X_train.shape
    w = np.zeros(D, dtype=np.float64)
    b = 0.0

    for _ in range(epochs):
        y_hat = X_train @ w + b              # (N,)
        error = y_hat - y_train              # (N,)
        grad_w = (2.0 / N) * (X_train.T @ error)   # (D,)
        grad_b = (2.0 / N) * error.sum()           # scalar
        w -= lr * grad_w
        b -= lr * grad_b

    y_pred = X_test @ w + b
    return w, float(b), y_pred
```

## 3. 容易踩的坑

1. **常数系数 2/N 漏掉**：测试会因数值不一致挂掉。常数省掉只是改变了「有
   效学习率」，最终训练出的权重和参考实现差很多。
2. **`X.T @ error` 顺序写反**：`error @ X` 也能 broadcast 但结果完全不对。
   记住「梯度对权重求偏导，要把 X 转置」。
3. **`b` 类型**：题目要求返回 Python `float`，注意 `float(b)` 转一下。如果
   直接返回 numpy scalar 类型不一致也会有警告（这道题判分宽松，能通过）。
4. **就地修改 vs 重新赋值**：`w -= lr * grad_w` 和 `w = w - lr * grad_w`
   等价。但如果 `w` 是被 caller 传进来共享的，就地修改会污染外部状态 ——
   本题不存在这个问题。

## 4. 进阶：能不能不用循环？

理论上线性回归有闭式解：$w = (X^\top X)^{-1} X^\top y$（正规方程）。但本
题考的是「梯度下降流程」，所以必须迭代。如果允许闭式解，一行：

```python
w = np.linalg.solve(X.T @ X, X.T @ y)
```

闭式解在 $D$ 很大或者 $X^\top X$ 病态时不稳定，所以工程上仍偏好梯度下降。
