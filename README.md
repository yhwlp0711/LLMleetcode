# ML LeetCode

> 面向 **ML / LLM 面试**的「手撕代码」自动判分系统 —— 手写线性回归、MHA、RoPE、KV Cache、Beam Search…… 写完自动评分，附中文解析。

<p>
  <img src="https://img.shields.io/badge/python-%E2%89%A53.10-blue" alt="python">
  <img src="https://img.shields.io/badge/PyTorch-%E2%89%A52.0-ee4c2c" alt="pytorch">
  <img src="https://img.shields.io/badge/题库-36_道-brightgreen" alt="problems">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="license">
</p>

支持 **Web UI**（推荐）和 **CLI** 两种使用方式，全部判分在本地执行，无需 Docker 或远程服务。

## 快速开始

```bash
git clone https://github.com/yhwlp0711/LLMleetcode.git
cd LLMleetcode
pip install -e ".[web]"     # 安装核心 + Web UI 依赖（推荐虚拟环境）

mlleetcode ui               # 启动 Web UI，自动打开 http://localhost:8000
```

想用命令行：

```bash
mlleetcode list             # 看题库
mlleetcode show mha         # 看题面
mlleetcode start mha        # 拷 starter 到 workspace/
mlleetcode submit workspace/pytorch__llm__mha.py   # 提交判分
mlleetcode solution mha     # 看中文解析 + 参考答案
```

## Web UI

- 题目列表：按分类分组、搜索过滤、**通过状态与进度**一目了然
- 题面渲染：Markdown + LaTeX 数学公式
- Monaco Editor：VS Code 同款编辑器，Python 语法高亮
- 一键提交：实时判分（per-case PASS/FAIL + diff）
- 参考解析 + 答案，代码一键复制
- 代码自动保存，刷新不丢失

## CLI 命令

| 命令 | 说明 |
|---|---|
| `mlleetcode list [前缀]` | 列出所有题，可按 dotted 前缀过滤（如 `pytorch.llm`） |
| `mlleetcode show <id>` | 渲染题面 |
| `mlleetcode solution <id>` | 中文解析 + 参考代码 |
| `mlleetcode start <id> [--force]` | 拷贝 starter 到 workspace/ |
| `mlleetcode submit <path> [-p <id>]` | 提交判分 |
| `mlleetcode verify [<id\|prefix>]` | 自检（跑参考实现，应 100/100） |
| `mlleetcode ui [--port N]` | 启动 Web UI |

## 题库

> 完整清单跑 `mlleetcode list` 即可查看。只想做 NumPy 题不装 PyTorch 也行（pytorch 题提交时会提示缺依赖）。

| 分类 | 内容 |
|---|---|
| **NumPy 基础** | 广播 / argmax / 滑窗 / 手写 matmul |
| **NumPy 经典 ML** | 线性 & 逻辑回归 / KMeans / KNN / PCA / AUC-ROC |
| **PyTorch 基础** | tensor 操作 / autograd |
| **PyTorch nn** | 激活函数（SiLU/GELU）/ 门控激活（SwiGLU/GeGLU）/ 数值稳定 sigmoid & softmax / LayerNorm / RMSNorm / 交叉熵 & BCE & KL 散度 |
| **PyTorch LLM（核心）** | causal mask / 位置编码 / SDPA / MHA / GQA / KV Cache / RoPE / 采样 / Greedy Decode / Beam Search / SwiGLU FFN / Transformer Block / DPO & PPO & GRPO loss |

<details>
<summary>展开完整题库（36 道）</summary>

| 题目 | 难度 | 考点 |
|---|---|---|
| `numpy.basics.broadcasting` | easy | 广播与外积、列归一化、按行缩放 |
| `numpy.basics.argmax_along_axis` | easy | 不用 `np.argmax` 自己实现 |
| `numpy.basics.sliding_window` | medium | 滑窗视图、移动平均、一维卷积 |
| `numpy.basics.matmul_manual` | medium | 手写 matmul / 转置 / batched matmul |
| `numpy.ml.linear_regression` | easy | 手算梯度 + 批 GD |
| `numpy.ml.logistic_regression` | easy | 数值稳定 sigmoid + BCE |
| `numpy.ml.kmeans` | medium | Lloyd 算法、空簇处理 |
| `numpy.ml.knn` | easy | 向量化距离矩阵 + 投票 |
| `numpy.ml.pca` | medium | SVD + 主成分 + 符号统一 |
| `numpy.ml.auc_roc` | medium | ROC 曲线与 AUC 计算 |
| `pytorch.basics.tensor_ops` | easy | flatten / softmax / pairwise dist / top-k |
| `pytorch.basics.autograd_basics` | easy | `.backward()`、数值 Jacobian、SGD |
| `pytorch.ml.linear_regression` | easy | autograd 训练循环 |
| `pytorch.nn.activations` | easy | SiLU、GELU-exact、GELU-tanh |
| `pytorch.nn.gated_activations` | easy | SwiGLU、GeGLU（门控） |
| `pytorch.nn.numeric_activations` | easy | 数值稳定 sigmoid / softmax |
| `pytorch.nn.layernorm` | medium | LayerNorm（weight + bias） |
| `pytorch.nn.rmsnorm` | easy | RMSNorm（LLaMA 风格） |
| `pytorch.nn.cross_entropy` | easy | 稳定 CE from logits + ignore_index |
| `pytorch.nn.bce_with_logits` | easy | 稳定 BCE with logits |
| `pytorch.nn.kl_divergence` | easy | forward KL from logits |
| `pytorch.llm.causal_mask` | easy | 因果 + padding mask 构造 |
| `pytorch.llm.sinusoidal_pe` | easy | 经典 sin/cos 位置编码 |
| `pytorch.llm.scaled_dot_product_attention` | easy | 注意力核心算子 |
| `pytorch.llm.mha` | medium | 多头注意力（纯函数） |
| `pytorch.llm.gqa` | medium | Grouped-Query Attention |
| `pytorch.llm.kv_cache` | medium | 带 KV Cache 的 SDPA |
| `pytorch.llm.rope` | medium | RoPE：建表 + 旋转 |
| `pytorch.llm.top_k_top_p_sampling` | medium | 温度 + top-k + nucleus |
| `pytorch.llm.greedy_decode` | easy | Greedy 解码 |
| `pytorch.llm.beam_search` | medium | Beam Search + 长度归一化 |
| `pytorch.llm.swiglu_ffn` | medium | LLaMA 风格 FFN |
| `pytorch.llm.transformer_block` | hard | 完整 LLaMA block（集成题） |
| `pytorch.llm.dpo_loss` | medium | DPO 偏好损失（logsigmoid） |
| `pytorch.llm.ppo_clip_loss` | hard | GAE 优势 + PPO 裁剪代理损失 |
| `pytorch.llm.grpo_loss` | medium | GRPO 组内优势 + PPO 裁剪 |

</details>

## 判分原理

- **数值对比**：`torch.allclose` 语义，支持 ndarray / Tensor / scalar
- **自定义判定**：shape 检查、init 分布检查等
- **种子固定**：`random` / `numpy` / `torch` 全栈 seed，保证可复现
- **设备自动选择**：cuda → mps → cpu，MPS 自动放宽容差
- **运行时限**：每题 < 2 秒，超时警告

## 项目结构

```
├── mlleetcode/          # 核心框架（cli / judge / registry / report / server / utils）
├── problems/            # 题库，每题 5 文件（numpy/ 与 pytorch/ 下分类）
├── web/                 # React 前端（dist/ 已提交，用户无需装 node）
├── workspace/           # 用户写代码的地方
├── tests/               # 框架测试
└── docs/AUTHORING.md    # 出题指南
```

## 开发 / 加题

```bash
pip install -e ".[web]"
pytest -q                 # 框架测试
mlleetcode verify         # 全题自检（应 100/100）

# 前端开发（需要 node）
cd web && npm install && npm run dev             # dev server :5173
uvicorn mlleetcode.server.app:app --reload --port 8000   # 后端单独起
```

加题参考 [docs/AUTHORING.md](docs/AUTHORING.md)：每道题 5 个文件（`meta.yaml` + `README.md` + `starter.py` + `solution.py` + `test_cases.py`），跑 `mlleetcode verify <id>` 确认 100/100 即可。

## License

MIT
