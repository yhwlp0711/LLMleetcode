# 解题思路：手撕线性回归

## 一句话思路

线性回归（Linear Regression）用一条「超平面」$\hat{y} = Xw + b$ 去拟合数据，
目标是最小化均方误差（MSE）。这题的核心就是**手写批量梯度下降**（Batch
Gradient Descent）循环：前向算预测 → 算梯度 → 更新参数，重复 `epochs` 轮。

## 从直觉到公式

### 模型与损失

线性模型：$\hat{y} = Xw + b$，其中 $w$ 是 `(D,)` 权重，$b$ 是标量偏置。

均方误差（MSE, Mean Squared Error）：

$$L = \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)^2$$

### 梯度怎么来的？

记误差向量 $e = \hat{y} - y$（shape `(N,)`），对 $w$ 和 $b$ 分别求偏导：

$$\nabla_w L = \frac{2}{N} X^\top e$$

$$\nabla_b L = \frac{2}{N} \sum_i e_i$$

直觉：$\nabla_w$ 的第 $j$ 分量 = 误差和第 $j$ 个特征的「内积」再乘常数；
$\nabla_b$ 就是误差的平均值 × 2。

### 更新规则

每轮把参数往梯度反方向走一小步：

$$w \leftarrow w - \text{lr} \cdot \nabla_w, \quad b \leftarrow b - \text{lr} \cdot \nabla_b$$

重复 `epochs` 轮后，用最终的 `(w, b)` 在测试集上预测。

## 参考实现

```python
import numpy as np

def fit_predict(X_train, y_train, X_test, *, lr, epochs):
    N, D = X_train.shape
    w = np.zeros(D, dtype=np.float64)
    b = 0.0

    for _ in range(epochs):
        y_hat = X_train @ w + b                         # 前向
        error = y_hat - y_train                         # 误差 (N,)
        grad_w = (2.0 / N) * (X_train.T @ error)       # (D,)
        grad_b = (2.0 / N) * error.sum()                # scalar
        w -= lr * grad_w
        b -= lr * grad_b

    y_pred = X_test @ w + b
    return w, float(b), y_pred
```

## 关键点

1. **常数 `2/N` 不能省**：省掉等价于学习率减半。本题判分器逐位比对数值
   （`atol=1e-6`），常数写错梯度大小就不对、权重收敛值也跟着偏。

2. **`X_train.T @ error` 就是 $\sum_i e_i x_i$**：`X_train` 是 `(N, D)`，
   转置后 `(D, N)` 和 error `(N,)` 做矩阵乘得 `(D,)`。一行代替了 D 次循环
   求内积。

3. **初始化全零是安全的**：线性回归的 MSE 是凸函数，不像神经网络有对称性
   问题，从全零出发梯度下降一定能收敛到全局最优。

4. **延伸**：如果允许闭式解，一行 `w = np.linalg.solve(X.T @ X, X.T @ y)`
   就搞定（正规方程 normal equation）。但实际中 D 很大或矩阵病态时闭式解
   不稳定/太慢，所以梯度下降仍然是主流。逻辑回归（见 `numpy.ml.logistic_regression`）
   用同样的梯度下降框架，只是换了损失和激活函数。
