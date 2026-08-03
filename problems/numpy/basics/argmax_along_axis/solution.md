# 解题思路：按轴 argmax

## 一句话思路

手写 `np.argmax(x, axis=axis)`——沿指定轴找每条「线」上最大值的索引。难点
不在「找最大」，而在**怎么处理任意维度 + 任意轴**。核心套路：先用 `moveaxis`
把目标轴搬到最后，压平成 `(B, D)` 的二维形式，再做一趟向量化扫描即可。

## 拆解思路

### 把「任意轴」变成「最后一维」

沿任意轴做操作很麻烦，因为轴位置不固定。关键观察：**只要把目标轴挪到最后**，
剩余维度合并成一个「批量」维，就能统一处理。

- `np.moveaxis(x, axis, -1)` —— 得到形状 `(..., D_axis)`。
- `reshape(-1, D)` —— 前面所有维度压平，变成规整的 `(B, D)` 二维数组。

现在每一行就是「要在其中找最大下标」的一条数据。

### 向量化扫描：维护「当前最大」

按列从左向右扫（`j = 1, 2, ..., D-1`），对所有行同时维护 `best_val` 和
`best_idx`。每来一列，比较是否比当前最大更大：

$$\text{better}_r = (\text{col}_r > \text{bestval}_r)$$

用严格大于 `>`：并列时不更新，自然保留**最早出现**的下标——满足「并列取
最小索引」的要求。

### 还原形状

扫完后 `best_idx` 长度为 `B`。reshape 回 `moved.shape[:-1]`（去掉目标轴
后的形状），就和 `np.argmax` 输出一致。

## 参考实现

```python
import numpy as np

def argmax_along_axis(x: np.ndarray, axis: int) -> np.ndarray:
    moved = np.moveaxis(x, axis, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    N, D = flat.shape

    best_idx = np.zeros(N, dtype=np.int64)
    best_val = flat[:, 0].copy()
    for j in range(1, D):
        col = flat[:, j]
        better = col > best_val             # 严格 >，并列保留更早下标
        best_val = np.where(better, col, best_val)
        best_idx = np.where(better, j, best_idx)

    return best_idx.reshape(moved.shape[:-1])
```

## 关键点

1. **`moveaxis` + `reshape` 是处理「任意轴」的通用招式**：把要操作的轴搬到
   固定位置，其余维度压平成批量维。很多 NumPy 手写题（归约 reduce、排序、
   扫描）都能用这招把「N 维 + 任意轴」化简成「二维」。

2. **严格 `>` 控制 tie-breaking**：用 `>` 时，后面出现的相等值不会覆盖前面的
   下标，天然满足「并列取最小索引」。若换成 `>=` 则取到最后一个并列位置。
   这是靠比较符号就能完成 tie-breaking 的小巧思。

3. **循环只沿 `D` 走，不沿样本走**。外层 `for j in range(D)` 每次用
   `np.where` 一次性更新所有行——这是向量化（vectorize）的关键，比「对每个
   样本单独循环找最大」快得多。

4. **延伸**：把 `>` 换成 `<` 就是 `argmin`。KMeans 里给样本分配最近质心
   （见 `numpy.ml.kmeans`）本质就是沿某个轴做 argmin，思路完全相通。
