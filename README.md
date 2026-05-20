# mlleetcode

面向 **ML / LLM 面试**的「手撕代码」自动判分系统 —— 手写线性回归、Scaled
Dot-Product Attention、MHA、RoPE、SwiGLU FFN…… 写完后由命令行自动评分。

无 GUI，代码在自己喜欢的 IDE 里写，判分一条命令搞定。

## 快速开始

```bash
# 1. 安装（推荐 uv，pip 也行）
uv venv --python 3.12 .venv
uv pip install -e .

# 2. 看题库列表
mlleetcode list

# 3. 看具体题目
mlleetcode show numpy.ml.linear_regression
mlleetcode show sdpa                # 缩写也行

# 4. 把 starter 拷到 workspace/ 开始写
mlleetcode start sdpa
$EDITOR workspace/pytorch__llm__scaled_dot_product_attention.py

# 5. 提交判分
mlleetcode submit workspace/pytorch__llm__scaled_dot_product_attention.py

# 6. 看不会？看参考答案 + 中文解析
mlleetcode solution sdpa
```

## 命令一览

| 命令 | 说明 |
|---|---|
| `mlleetcode list [前缀]` | 列出所有题，可按 dotted 前缀过滤（如 `pytorch.llm`）。 |
| `mlleetcode show <id>` | 用 markdown 渲染该题的题面（README）。 |
| `mlleetcode solution <id>` | 显示该题的中文解析（solution.md）与参考代码（solution.py）。 |
| `mlleetcode start <id> [--force]` | 把 `starter.py` 拷到 `workspace/<flat-id>.py` 让你编辑。 |
| `mlleetcode submit <path> [-p <id>] [--seed N]` | 提交判分。题号不显式给时由文件名推断。 |
| `mlleetcode verify [<id\|prefix>]` | 自检：用参考实现跑判分，应该 100/100。 |

## 题号约定

每道题位于 `problems/<框架>/<分类>/<slug>/` 目录，完整 ID 就是 dotted path：

```
problems/numpy/ml/linear_regression/                  →  numpy.ml.linear_regression
problems/pytorch/llm/scaled_dot_product_attention/   →  pytorch.llm.scaled_dot_product_attention
```

CLI 接受以下任意写法（按唯一性匹配）：

- **完整 ID**：`numpy.ml.linear_regression`
- **后缀**：`ml.linear_regression`（须唯一）
- **叶子 slug**：`linear_regression`（须唯一；当 numpy 和 pytorch 都有
  `linear_regression` 时不唯一，要至少加一段）
- **段内前缀**：`numpy.ml.lin` 匹配 `linear_regression`
- **子串**：`numpy.ml.linear_regress`
- **首字母缩写**：`sdpa` 匹配 `scaled_dot_product_attention`

workspace 文件名把路径用 `__` 拉平：
`workspace/numpy__ml__linear_regression.py`。

## 判分原理

- **数值判分。** 大多数用例直接对比函数输出与参考实现，按 `torch.allclose`
  语义（`atol` / `rtol` 可按题配置）。`numpy.ndarray`、`torch.Tensor`、
  Python scalar / list 都支持。
- **自定义判定。** TestCase 的 `runner` 也可以直接返回 `CompareResult`，
  用于检查参数 shape、init 分布等需要专门逻辑的场景。
- **不限制实现方式。** 想用什么库都行，只看结果对不对。题面里会说明
  「手撕」的精神，自觉遵守即可。
- **种子固定。** 每个用例前都会 `set_seed`（`random` / `numpy` / `torch`
  全栈，默认 42），保证可复现。
- **设备自动选择。** 优先 `cuda:0`，其次 `mps`，最后 `cpu`。MPS 上判分容差
  会自动放宽一点（fp32 精度略低）。
- **超时控制。** 每题在 `meta.yaml` 里设置 per-case 超时。

## 题库（共 27 道，全部自检通过）

### NumPy 基础
- `numpy.basics.broadcasting` — 广播与外积、列归一化、按行缩放
- `numpy.basics.argmax_along_axis` — 不调用 `np.argmax` 自己实现按轴 argmax
- `numpy.basics.sliding_window` — 滑窗视图、移动平均、一维卷积
- `numpy.basics.matmul_manual` — 手写 matmul / 转置 / batched matmul

### NumPy 经典 ML
- `numpy.ml.linear_regression` — 手算梯度 + 批 GD
- `numpy.ml.logistic_regression` — 数值稳定 sigmoid + BCE
- `numpy.ml.kmeans` — Lloyd 算法（含空簇处理）
- `numpy.ml.knn` — KNN 分类（向量化距离矩阵 + 投票）
- `numpy.ml.pca` — SVD + 主成分 + 符号统一
- `numpy.ml.auc_roc` — ROC 曲线与 AUC 计算

### PyTorch 基础
- `pytorch.basics.tensor_ops` — flatten / softmax / pairwise distance / top-k
- `pytorch.basics.autograd_basics` — `.backward()`、数值 Jacobian、SGD 循环

### PyTorch ML（autograd 版）
- `pytorch.ml.linear_regression` — 同一道题，用 autograd 训练

### PyTorch nn.Module / 激活函数
- `pytorch.nn.activations` — SiLU、GELU（精确 + tanh 近似）、SwiGLU、GeGLU
- `pytorch.nn.layernorm` — 带 weight + bias 的 LayerNorm 模块
- `pytorch.nn.rmsnorm` — LLaMA 风格 RMSNorm 模块

### PyTorch LLM
- `pytorch.llm.causal_mask` — 因果 + padding mask 构造
- `pytorch.llm.sinusoidal_pe` — 经典 Sinusoidal 位置编码
- `pytorch.llm.scaled_dot_product_attention` — 注意力核心算子
- `pytorch.llm.mha` — 多头注意力（纯函数版，权重作为入参）
- `pytorch.llm.gqa` — Grouped-Query Attention（LLaMA-2/3、Mistral 标配）
- `pytorch.llm.kv_cache` — 带 KV cache 的 SDPA（推理优化）
- `pytorch.llm.rope` — Rotary Position Embedding：建表 + 应用
- `pytorch.llm.top_k_top_p_sampling` — 温度 + top-k + nucleus 过滤
- `pytorch.llm.greedy_beam_search` — Greedy + Beam Search 解码
- `pytorch.llm.swiglu_ffn` — LLaMA 风格门控 FFN
- `pytorch.llm.transformer_block` — LLaMA-style 完整 block（集成题）

## 出题指南

如何添加新题、两种判分模式（Pattern A 算子题 / Pattern B 模块题）、避坑指南
（dropout、in-place ops 等）见 [docs/AUTHORING.md](docs/AUTHORING.md)。

## 开发

```bash
uv pip install -e .
pytest -q                 # 框架自身测试
mlleetcode verify         # 跑所有题的参考解，全 ACCEPTED 才算 OK
```
