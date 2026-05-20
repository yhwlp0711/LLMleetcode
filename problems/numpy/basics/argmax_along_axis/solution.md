# 解题思路：按轴 argmax

## 分析

`np.argmax(x, axis=axis)` 在 N 维数组上做的事情，等价于：

1. 把目标轴搬到最后，得到 shape `(..., D)`。
2. 把前面所有维 flatten 成一个 batch 维，变成 `(B, D)`。
3. 对每行求最大值的列索引 → 长度 `B` 的一维向量。
4. reshape 回 `(...,)`（即去掉目标轴的形状）。

第 3 步禁用 `np.argmax`，怎么办？两种向量化思路：

### 思路 A（推荐）：扫描更新

对每一列 `j = 1, 2, ..., D-1`，比较「当前列的值」和「目前为止的最大值」，
用 `np.where` 同时更新「最大值」与「对应索引」。这相当于把 `argmax` 的 reduce
循环手动展开。**注意**：用严格大于 `>`，这样并列时保留更小的索引（题目要求）。

### 思路 B：flat-argmax 技巧

每行加一个「列索引」当 tie-breaker：`x[b, j] * D + (D - 1 - j)`，然后用
... 但还是要求 `argmax`，反而更绕。**思路 A 更直白。**

## 参考实现

```python
def argmax_along_axis(x, axis):
    moved = np.moveaxis(x, axis, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    N, D = flat.shape

    best_idx = np.zeros(N, dtype=np.int64)
    best_val = flat[:, 0].copy()
    for j in range(1, D):
        col = flat[:, j]
        better = col > best_val          # 严格大于 → 并列时保留更小索引
        best_val = np.where(better, col, best_val)
        best_idx = np.where(better, j, best_idx)

    return best_idx.reshape(moved.shape[:-1])
```

## 关键点

1. **`np.moveaxis(x, src, dst)`** 把指定轴搬到目标位置；比 `transpose` 更易读。
2. **`reshape(-1, D)`** 把所有 batch 维 flatten。`-1` 让 NumPy 自动推断。
3. **`np.where(cond, a, b)`** 是向量化的三元运算符。
4. **循环次数是 `D`**（轴长），而不是数据规模；只要轴不太长就完全够快。
5. **并列处理**：题目说"取最小索引"，所以用 `>`（非 `>=`）。如果反过来要
   "最大索引"，把 `>` 换成 `>=`。
