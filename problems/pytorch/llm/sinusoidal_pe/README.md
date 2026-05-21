# Sinusoidal Position Encoding

实现 Transformer 论文里的经典正弦/余弦位置编码（PE）。RoPE 出现前 6 年的
位置编码方案，BERT、GPT-2 等模型都用它的变体。

## 函数签名

```python
def build_sinusoidal_pe(seq_len: int, d_model: int) -> torch.Tensor:
    """返回 shape (seq_len, d_model) 的位置编码张量。"""
```

## 公式

对位置 $pos$ 和维度 $i$：

$$\text{PE}[pos, 2i] = \sin\!\bigl(pos / 10000^{2i / d\_{\text{model}}}\bigr)$$

$$\text{PE}[pos, 2i+1] = \cos\!\bigl(pos / 10000^{2i / d\_{\text{model}}}\bigr)$$

即：**偶数维放 sin，奇数维放 cos**，频率 $\theta\\_i = 1/10000^{2i/d\\_{\text{model}}}$。

## 说明

- `d_model` 一定是偶数。
- 输出 `torch.float32`。
- 容差 `atol=1e-5`。
- 一次性算出整个表，不要循环；用广播 + 一次 `sin/cos` 调用。
