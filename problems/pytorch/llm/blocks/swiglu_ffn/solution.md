# 解题思路：SwiGLU FFN

## 一句话思路

现代 LLM（LLaMA / Mistral / Gemma）的前馈网络（feed-forward network, FFN）用的是
**门控（gated）结构**：一路做 SiLU 激活当「门」，另一路当「内容」，两路逐元素相
乘后再投影回去。所以核心就是一行 `down_proj(silu(gate_proj(x)) * up_proj(x))`，用
了 3 个线性层。

## 拆解思路

### 从 2 个矩阵到 3 个矩阵

原始 Transformer 的 FFN 是 `Linear → ReLU → Linear`，两个矩阵：

```
out = W2 @ activation(W1 @ x)
```

SwiGLU 把它换成**门控**形式：一路 `gate_proj(x)` 过 SiLU 激活当作「开关」，另一路
`up_proj(x)` 是「原始内容」，两者逐元素相乘（gate 决定内容里每个通道放行多少），最
后 `down_proj` 投影回 `d_model`：

$$\text{FFN}(x) = W_{\text{down}}\bigl(\text{SiLU}(W_{\text{gate}}\,x) \odot W_{\text{up}}\,x\bigr)$$

其中 SiLU（也叫 Swish）是 $\text{SiLU}(z) = z \cdot \sigma(z)$，一个平滑、可正可负
的激活。门控让网络能自适应地「筛选」哪些特征通过，实践中比普通 ReLU 效果更好。

### 维度怎么走

输入 `x` 是 `(..., d_model)`：

1. `gate_proj(x)` → `(..., d_ff)`
2. `up_proj(x)` → `(..., d_ff)`
3. `silu(gate) * up` → `(..., d_ff)`（逐元素）
4. `down_proj(...)` → `(..., d_model)`

前面所有维度（batch、序列）保持不变，符合 FFN「逐 token 独立」的语义。

## 参考实现

```python
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)   # 门
        self.up_proj   = nn.Linear(d_model, d_ff, bias=False)   # 内容
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)   # 投影回去

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

## 关键点

1. **三个线性层的名字必须叫 `gate_proj` / `up_proj` / `down_proj`**。判分会用
   `load_state_dict` 把参考权重注入到这些名字上，改成 `linear1` 之类会加载失败。
   这和 LLaMA 官方命名一致（早期叫 `w1/w2/w3`）。

2. **都没有 bias（`bias=False`）**。这是 LLaMA 系列的选择——实验表明 FFN 的 bias
   对效果没明显帮助，去掉能省参数和带宽。

3. **`silu(gate) * up` 是逐元素相乘**，两者都是 `(..., d_ff)`，对齐相乘后 gate
   起「逐通道门控」的作用。SiLU 可以用 `F.silu`，等价于 `z * torch.sigmoid(z)`。

4. **`d_ff` 通常约等于 `2.67 * d_model`**。门控 FFN 有 3 个矩阵而非 2 个，为了让
   总参数量和经典 FFN 差不多，需要把隐藏维 `d_ff` 从 GPT 风格的 `4 * d_model` 缩
   到约 `(8/3) * d_model`。

5. **延伸**：不加 dropout 是为了让前向输出确定、便于精确数值对比。把这个 FFN 和
   RMSNorm、注意力拼成完整的 decoder 层，就是
   `pytorch.llm.blocks.transformer_block`。
