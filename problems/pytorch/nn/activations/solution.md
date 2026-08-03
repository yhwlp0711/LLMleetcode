# 解题思路：激活函数（SiLU / GELU）

## 一句话思路

三个函数都是现代 Transformer / LLM 常用的激活函数（activation function），本质是
把课本公式**照抄成张量运算**。真正值得理解的是「为什么现代模型偏爱这些平滑激活，
而不是老的 ReLU」，以及 GELU 为什么有精确版和 tanh 近似版两种写法。

## 从直觉到公式

### SiLU（也叫 Swish）

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

可以把它理解成「自带门控的 ReLU」：`sigmoid(x)` 是一个 0~1 之间的软开关，`x` 很大
时开关趋近 1（几乎放行原值），`x` 很负时趋近 0（几乎堵死）。和 ReLU 硬生生在 0 处
截断不同，SiLU 处处平滑可微，负半轴也有一点点非零梯度，缓解了**神经元死亡（dead
neuron）**——ReLU 在 `x<0` 时梯度恒为 0，一旦落进去就再也学不动。

### GELU（精确版）

$$\text{GELU}(x) = x \cdot \Phi(x) = \frac{x}{2}\bigl(1 + \operatorname{erf}(x / \sqrt{2})\bigr)$$

$\Phi(x)$ 是标准正态分布的累积分布函数，代表「标准正态随机变量 ≤ x 的概率」。所以
GELU 相当于「按输入的大小，用一个概率去缩放它」：输入越大越可能被完整保留。它用误
差函数（error function）`torch.erf` 就能精确算出，PyTorch 内置无需自己实现。

### GELU（tanh 近似版）

$$\text{GELU}_{\tanh}(x) = \frac{x}{2}\bigl(1 + \tanh\bigl[\sqrt{2/\pi}\,(x + 0.044715\,x^3)\bigr]\bigr)$$

这是原论文给的近似式，用 `tanh` 加一个三次多项式去逼近上面的 `erf`。历史原因是早期
硬件 / 低精度算子里 `erf` 实现昂贵，`tanh` 更快。

## 参考实现

```python
from math import pi, sqrt

def silu(x):
    return x * torch.sigmoid(x)

def gelu_exact(x):
    return 0.5 * x * (1.0 + torch.erf(x / sqrt(2.0)))   # 用误差函数精确算

def gelu_tanh(x):
    c = sqrt(2.0 / pi)
    return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * x.pow(3))))
```

## 关键点

1. **为什么现代模型爱用平滑激活**：ReLU 在负半轴梯度为 0，容易造成神经元死亡；
   SiLU / GELU 处处可微、负半轴留有微弱梯度，训练更稳、表达更细腻。LLaMA、Mistral、
   Gemma 用 SiLU，BERT、GPT 用 GELU。

2. **精确版和 tanh 近似不能混用**：两者数值上有微小差异。加载别人的预训练权重时，
   必须和它训练时用的那一版对齐，否则前向输出对不上。GPT-2/3 训练时用的就是 tanh 近
   似版，所以复现它们要用 `gelu_tanh`。

3. **`torch.erf` 直接可用**：误差函数是 GELU 精确版的核心，PyTorch 把它当基础算子内
   置了，`x.pow(3)` 和 `x ** 3` 等价，前者语义更清楚。

4. **延伸**：把这些激活作用在「门」张量上、再逐元素乘另一个张量，就得到门控变
   体 SwiGLU / GeGLU，是现代 LLM 前馈网络的标配——见 `pytorch.nn.gated_activations`。
   sigmoid 本身的数值稳定实现见 `pytorch.nn.numeric_activations`。
