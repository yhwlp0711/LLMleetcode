# 解题思路：Sinusoidal Position Encoding

## 公式回顾

$$\text{PE}[pos, 2i]   = \sin(pos \cdot \theta_i),\quad
  \text{PE}[pos, 2i+1] = \cos(pos \cdot \theta_i),\quad
  \theta_i = 10000^{-2i / d_{\text{model}}}$$

## 参考实现

```python
def build_sinusoidal_pe(seq_len, d_model):
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, d_model, 2, dtype=torch.float32) / d_model))
    pos = torch.arange(seq_len, dtype=torch.float32)
    angles = pos[:, None] * inv_freq[None, :]    # (seq_len, d_model/2)

    pe = torch.zeros(seq_len, d_model, dtype=torch.float32)
    pe[:, 0::2] = angles.sin()
    pe[:, 1::2] = angles.cos()
    return pe
```

## 三个关键操作

### 1. `torch.arange(0, d_model, 2)` 取偶数索引

得到 `[0, 2, 4, ..., d_model-2]`，长度 `d_model/2`。每个 `2i` 对应一个
独立的频率 `theta_i`。

### 2. 外积构造角度矩阵

`pos[:, None] * inv_freq[None, :]`：
- `pos[:, None]` 是 `(seq_len, 1)`
- `inv_freq[None, :]` 是 `(1, d_model/2)`
- 广播相乘 → `(seq_len, d_model/2)`

`angles[m, i] = m * theta_i`，每个位置 × 每个频率。

### 3. 切片赋值 `[:, 0::2]` 和 `[:, 1::2]`

把 sin 填到偶数列，cos 填到奇数列。`pe[:, 0::2] = angles.sin()` 这种切片
赋值是 PyTorch 标准操作，shape 自动对齐 `(seq_len, d_model/2)`。

## 跟 RoPE 的对比

| | Sinusoidal PE | RoPE |
|---|---|---|
| 形式 | 加性（`x + pe`） | 乘性（旋转）|
| 作用对象 | 输入 embedding | Q / K 张量 |
| 相对位置 | 隐式（在 attention 计算中体现）| 显式（旋转矩阵的乘积自然得到相对距离）|
| 外推性 | 一般（训练时见过的位置外才差）| 更好 |
| 应用 | BERT, GPT-2 | LLaMA, Mistral, Qwen |

虽然 RoPE 已是主流，sinusoidal PE 仍然出现在很多面试题里 —— 因为它的数
学性质（不同频率的正交基）是理解 attention「为什么能编码相对位置」的入
口。

## 「位置 0 = [0, 1, 0, 1, ...]」的属性测试

题目里有一个 property 测试：位置 0 的 PE 应该是 `[0, 1, 0, 1, ...]`。
因为 `pos=0` 时所有 `angles = 0`，所以 `sin(0)=0, cos(0)=1`。

这种属性测试比"数值对比"更能捕捉「公式记反」的 bug —— 如果你把 sin/cos
位置搞反，数值对比会因为参考实现也错而通过；但 property 测试有独立的预
期值，会暴露 bug。

## 为什么 base 是 10000？

来自论文，作者表示：「我们选择这个值是因为它对我们的训练序列长度（512）
来说足够长，能让最低频率覆盖完整周期」。10000 没什么神奇，是个工程经验
值。RoPE 通常也用 10000；LLaMA-3 等模型为了支持长上下文（128k+）把
base 调大到 50w 甚至 100w。
