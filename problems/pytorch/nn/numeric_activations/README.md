# 数值稳定的 Sigmoid / Softmax

手写 sigmoid 和 softmax，重点是**数值稳定性**——朴素实现会在极端输入下溢出/上溢。
这是 ML 面试高频考点。

## 待实现函数

### 1. `sigmoid(x)`

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

要求**数值稳定**：`exp` 遇到大参数会溢出成 `inf`（`x = 1000` → `exp(1000) = inf`）。
朴素写法 `exp(x)/(1+exp(x))` 在大正值时会得到 `inf/inf = nan`。
提示：对 $x \geq 0$ 和 $x < 0$ 分别处理——利用恒等式 $\frac{1}{1+e^{-x}} = \frac{e^{x}}{1+e^{x}}$，
让 `exp` 的参数永远 $\leq 0$，输出始终有限。

### 2. `softmax(x, dim)`

$$\text{softmax}(x\_i) = \frac{e^{x\_i - \max(x)}}{\sum\_j e^{x\_j - \max(x)}}$$

沿指定 `dim` 做 softmax。**必须数值稳定**：先减去该维的 max 再 exp
（softmax 平移不变，减常数不改变结果，但避免了 `exp` 上溢）。

```python
def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    ...
```

## 说明

- 输入是 `torch.float32`。**禁止用** `torch.sigmoid` / `F.softmax`，自己按公式实现。
- 会用极端值（如 `±1000`）测试稳定性。
- 容差 `atol=1e-6`。
