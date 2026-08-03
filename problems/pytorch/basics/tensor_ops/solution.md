# 解题思路：PyTorch 张量操作热身

## 一句话思路

五个小函数练的都是 PyTorch 张量的**向量化**基本功：怎么展平拼接、怎么写数值稳定
的 softmax、怎么用广播（broadcasting）一次性算出成对距离、怎么带掩码求均值、怎么
取 top-k。核心是「用张量运算代替 Python 循环」这个思维方式。

## 拆解思路

### 1. `flatten_and_concat`：展平后拼接

`reshape(-1)` 把任意形状压成一维；`torch.cat` 沿默认的 `dim=0` 拼接。相比
`view(-1)`，`reshape` 不要求张量在内存里连续（contiguous），更鲁棒——判分会同时用
连续和非连续输入测你。

### 2. `row_softmax`：数值稳定的按行 softmax

朴素写法 `exp(x) / exp(x).sum()` 有个大坑：`x` 里若有大数（比如 1000），
`exp(1000)` 直接溢出（overflow）变 `inf`，结果全崩。关键观察是 softmax 有**平移不
变性（shift invariance）**：给一行所有元素同时减去一个常数，结果不变。于是我们减
去每行的最大值，让所有指数都 ≤ 0，`exp` 落进 (0, 1] 安全区。

### 3. `pairwise_squared_distance`：用代数展开避免循环

要算 `x[i]` 与 `y[j]` 两两之间的平方距离。直接嵌套循环太慢，用这个恒等式拆开：

$$\|x_i - y_j\|^2 = \|x_i\|^2 - 2\,x_i^\top y_j + \|y_j\|^2$$

三项分别对应：`x` 每行的平方和（形状 `(N,1)`）、矩阵乘 `x @ yᵀ`（形状 `(N,M)`）、
`y` 每行的平方和（形状 `(1,M)`）。靠广播把 `(N,1)`、`(N,M)`、`(1,M)` 加到一起，一
步得到 `(N,M)` 的结果，不写一个 Python 循环。

### 4. `masked_mean`：只对有效位置求均值

思路是「把无效位置乘成 0，求和后除以有效位置的个数」。mask 是布尔的，先转成
`x` 的浮点类型才能参与算术；再 `unsqueeze(-1)` 补一个维度好和 `(B,T,D)` 广播。

### 5. `top_k_indices`：取前 k 大的索引

`torch.topk(..., sorted=True)` 直接返回按值降序排好的索引。它返回 `(values,
indices)`，我们只要第二个。CPU 上的实现是稳定的（并列时索引小的在前），正好满足题
目「并列时索引较小者优先」。

## 参考实现

```python
def flatten_and_concat(a, b):
    return torch.cat([a.reshape(-1), b.reshape(-1)])


def row_softmax(x):
    m = x.max(dim=-1, keepdim=True).values     # 每行最大值 (N, 1)
    e = (x - m).exp()                          # 减 max 后指数都 ≤ 0
    return e / e.sum(dim=-1, keepdim=True)


def pairwise_squared_distance(x, y):
    x2 = (x * x).sum(dim=-1, keepdim=True)                   # (N, 1)
    y2 = (y * y).sum(dim=-1, keepdim=True).transpose(0, 1)   # (1, M)
    xy = x @ y.transpose(0, 1)                               # (N, M)
    return (x2 - 2 * xy + y2).clamp(min=0.0)                 # 削掉浮点负值


def masked_mean(x, mask):
    m = mask.to(x.dtype).unsqueeze(-1)                       # (B, T, 1)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


def top_k_indices(scores, k):
    _, idx = torch.topk(scores, k=k, largest=True, sorted=True)
    return idx
```

## 关键点

1. **softmax 为什么要减 max**：这是最常见的数值稳定（numerical stability）技巧。
   `exp` 的指数只要 ≤ 0 就不会溢出，减去最大值恰好把最大的指数拉到 0，其余更小。
   `keepdim=True` 让归约（reduce）掉的维度保留成长度 1，才能和原张量做广播相减/相
   除。同样的技巧见 `pytorch.nn.numeric_activations`。

2. **成对距离的 `clamp(min=0)`**：数学上平方距离一定 ≥ 0，但用 `a²-2ab+b²` 展开算
   时，浮点误差可能让本该为 0 的项算出一个极小的负数，`clamp` 把它削回 0。

3. **广播 vs 显式扩张的取舍**：成对距离也能写成
   `((x[:,None,:] - y[None,:,:])**2).sum(-1)`，更直观，但会生成一个 `(N,M,D)` 的中
   间张量，`N`、`M`、`D` 都大时容易爆内存。代数展开法只留 `(N,M)`，省内存。

4. **`masked_mean` 的 `clamp(min=1.0)`**：万一某个 batch 全是无效位置，分母会是 0
   导致除零。虽然题目保证不会发生，加上它是稳妥的防御写法。

5. **延伸**：`row_softmax` 是很多损失函数的地基。把它换成 log 空间的稳定写法，就是
   交叉熵里用到的 log-softmax，见 `pytorch.nn.cross_entropy`。
