# 解题思路：SwiGLU FFN

## 参考实现

```python
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj   = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

`forward` 是经典的一行 LLaMA FFN。

## 维度对账

输入 `x`：`(..., d_model)`

| 步骤 | 操作 | 输出 shape |
|---|---|---|
| 1 | `gate_proj(x)` | `(..., d_ff)` |
| 2 | `up_proj(x)` | `(..., d_ff)` |
| 3 | `silu(gate) * up` | `(..., d_ff)`（逐元素） |
| 4 | `down_proj(...)` | `(..., d_model)` |

`...` 部分（前面所有维度）保持不变，所以 `(B, T, d_model)` 输入直接得到
`(B, T, d_model)` 输出，符合 FFN「逐 token 独立」的语义。

## 为什么是 3 个矩阵而不是 2 个？

经典 Transformer FFN：

```
out = W2 @ activation(W1 @ x + b1) + b2
# W1: (d_model, d_ff), W2: (d_ff, d_model)
# 总参数 ≈ 2 * d_model * d_ff
```

SwiGLU FFN：

```
out = W_down @ (silu(W_gate @ x) * (W_up @ x))
# 三个矩阵，总参数 ≈ 3 * d_model * d_ff
```

加了 50% 的参数，所以为了**保持参数量大致相等**，LLaMA 把 `d_ff` 从 `4 *
d_model`（GPT 风格）缩小到 `(8/3) * d_model ≈ 2.67 * d_model`。这样总参数
量基本一致，但训练效果更好。

## 关于 bias

`bias=False` 是 LLaMA 系列的选择 —— 实验显示 FFN 里的 bias 对效果没有
明显帮助，省掉减少参数量和带宽。这是 LLM 的工程经验，跟 BERT/T5 那种带
bias 的设计形成对比。

## 为什么不放 dropout？

题面里强调过：dropout 涉及 RNG，会让 forward 的输出不确定，难以做精确数
值对比。生产代码可能在 `down_proj` 后加一层 dropout，但本题让你练 FFN 算
法，dropout 是噪声层，强制 eval / 不加它，让判分聚焦在「核心计算对不对」。

## 参数命名规约

题目强制要求 `gate_proj` / `up_proj` / `down_proj` 这三个名字，因为判分
要 `load_state_dict` 注入参考权重。如果你写成 `linear1` / `linear2` /
`linear3`，judge 在加载权重时会失败。这种「命名约定」就像 LeetCode 强制
函数名叫 `twoSum` 一样 —— 是 contract，不是限制。

LLaMA 官方代码里的命名也是 `gate_proj` / `up_proj` / `down_proj`（早期版
本叫 `w1` / `w2` / `w3`，新版本统一了语义化命名），这道题跟它对齐。
