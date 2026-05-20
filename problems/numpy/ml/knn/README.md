# KNN 分类

实现 K-Nearest Neighbors 分类器。**不写训练**（KNN 无训练阶段），直接给
训练集和测试集，预测每个测试样本的类别。

## 函数签名

```python
def knn_predict(
    X_train: np.ndarray,   # (N, D) 训练特征
    y_train: np.ndarray,   # (N,)   训练标签 int64，取值 [0, num_classes)
    X_test:  np.ndarray,   # (M, D) 测试特征
    k: int,
    num_classes: int,
) -> np.ndarray:           # (M,)   预测标签 int64
```

## 算法

对每个测试样本：

1. 算它到所有训练样本的**欧氏距离平方**（不开根号节省计算，不影响排序）。
2. 取最近的 k 个训练样本。
3. 这 k 个样本投票：**出现次数最多的类别**胜出。
4. 并列时**取 id 较小的类别**。

## 说明

- 输入特征 `np.float64`；标签 `np.int64`。
- 输出 `np.int64`。
- **禁止用 sklearn / scipy**；只能用 NumPy。
- 性能上要求**向量化**计算距离矩阵 —— 一次性算完 `(M, N)`，不要循环。
  但 k 近邻的「投票」可以循环 M 次。
