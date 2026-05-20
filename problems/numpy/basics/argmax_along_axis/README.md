# 按轴 argmax

不调用 `np.argmax` / `np.argpartition`，自己实现「沿指定轴求最大值索引」的功能
（其他 NumPy 操作都可用）。这是练 strides / reshape / 高级索引的经典题。

## 函数签名

```python
def argmax_along_axis(x: np.ndarray, axis: int) -> np.ndarray:
    """
    参数:
        x:     N 维数组，shape 为 (..., D_axis, ...)
        axis:  合法的轴索引（可以是负数）

    返回:
        int64 数组，shape 等于把 `axis` 维去掉后的形状（与
        np.argmax(x, axis=axis) 相同）。元素是该轴上最大值的索引。
        如有并列最大值，取索引最小的。
    """
```

## 说明

- 禁用 `np.argmax` 与 `np.argpartition`。`np.argsort` 允许但没必要用。
- 提示：先 `np.moveaxis` 把目标轴搬到最后，把数组 reshape 成 `(B, D)` 这种
  二维形式，再用向量化的「逐列扫描更新当前最大」即可。
- 输入是 `np.float64`，输出必须是 `np.int64`。
