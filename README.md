# ML LeetCode

面向 **ML / LLM 面试**的「手撕代码」自动判分系统 —— 手写线性回归、MHA、
RoPE、KV Cache、Beam Search…… 写完后自动评分，附带中文解析。

支持 **CLI** 和 **Web UI** 两种使用方式。

![Web UI](web/src/assets/hero.png)

## 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/yhwlp0711/LLMleetcode.git
cd LLMleetcode

# 2. 安装（推荐用虚拟环境）
pip install -e ".[web]"        # 安装核心 + Web UI 依赖

# 3. 启动 Web UI（推荐）
mlleetcode ui                  # 自动打开浏览器 http://localhost:8000

# 或者用 CLI
mlleetcode list                # 看题库
mlleetcode show mha            # 看题面
mlleetcode start mha           # 拷 starter 到 workspace/
mlleetcode submit workspace/pytorch__llm__mha.py   # 提交判分
mlleetcode solution mha        # 看中文解析 + 参考答案
```

## 使用方式

### Web UI（推荐）

```bash
mlleetcode ui                  # 启动后自动打开浏览器
```

功能：
- 题目列表（按分类分组、搜索过滤）
- 题面渲染（Markdown + LaTeX 数学公式）
- Monaco Editor（VS Code 同款编辑器，Python 语法高亮）
- 一键提交 + 实时判分结果（per-case PASS/FAIL + diff）
- 参考解析 + 答案（代码可一键复制）
- 代码自动保存（刷新不丢失）

### CLI

| 命令 | 说明 |
|---|---|
| `mlleetcode list [前缀]` | 列出所有题，可按 dotted 前缀过滤（如 `pytorch.llm`） |
| `mlleetcode show <id>` | 渲染题面 |
| `mlleetcode solution <id>` | 中文解析 + 参考代码 |
| `mlleetcode start <id> [--force]` | 拷贝 starter 到 workspace/ |
| `mlleetcode submit <path> [-p <id>]` | 提交判分 |
| `mlleetcode verify [<id\|prefix>]` | 自检（跑参考实现，应该 100/100） |
| `mlleetcode ui [--port N]` | 启动 Web UI |

## 环境要求

- Python ≥ 3.10
- PyTorch ≥ 2.0（pytorch 类题目需要）
- NumPy ≥ 1.24

**所有判分都在你本地的 Python 环境执行**，不需要 Docker 或远程服务。
如果你只想做 NumPy 题，不装 PyTorch 也行（pytorch 题提交时会提示缺依赖）。

## 题库（27 道，全部验证通过）

### NumPy 基础
| 题目 | 难度 | 考点 |
|---|---|---|
| `numpy.basics.broadcasting` | easy | 广播与外积、列归一化、按行缩放 |
| `numpy.basics.argmax_along_axis` | easy | 不用 `np.argmax` 自己实现 |
| `numpy.basics.sliding_window` | medium | 滑窗视图、移动平均、一维卷积 |
| `numpy.basics.matmul_manual` | medium | 手写 matmul / 转置 / batched matmul |

### NumPy 经典 ML
| 题目 | 难度 | 考点 |
|---|---|---|
| `numpy.ml.linear_regression` | easy | 手算梯度 + 批 GD |
| `numpy.ml.logistic_regression` | easy | 数值稳定 sigmoid + BCE |
| `numpy.ml.kmeans` | medium | Lloyd 算法、空簇处理 |
| `numpy.ml.knn` | easy | 向量化距离矩阵 + 投票 |
| `numpy.ml.pca` | medium | SVD + 主成分 + 符号统一 |
| `numpy.ml.auc_roc` | medium | ROC 曲线与 AUC 计算 |

### PyTorch 基础
| 题目 | 难度 | 考点 |
|---|---|---|
| `pytorch.basics.tensor_ops` | easy | flatten / softmax / pairwise dist / top-k |
| `pytorch.basics.autograd_basics` | easy | `.backward()`、数值 Jacobian、SGD |

### PyTorch ML
| 题目 | 难度 | 考点 |
|---|---|---|
| `pytorch.ml.linear_regression` | easy | autograd 训练循环 |

### PyTorch nn.Module / 激活函数
| 题目 | 难度 | 考点 |
|---|---|---|
| `pytorch.nn.activations` | easy | SiLU、GELU、SwiGLU、GeGLU |
| `pytorch.nn.layernorm` | medium | LayerNorm（weight + bias） |
| `pytorch.nn.rmsnorm` | easy | RMSNorm（LLaMA 风格） |

### PyTorch LLM（核心）
| 题目 | 难度 | 考点 |
|---|---|---|
| `pytorch.llm.causal_mask` | easy | 因果 + padding mask 构造 |
| `pytorch.llm.sinusoidal_pe` | easy | 经典 sin/cos 位置编码 |
| `pytorch.llm.scaled_dot_product_attention` | easy | 注意力核心算子 |
| `pytorch.llm.mha` | medium | 多头注意力（纯函数） |
| `pytorch.llm.gqa` | medium | Grouped-Query Attention |
| `pytorch.llm.kv_cache` | medium | 带 KV Cache 的 SDPA |
| `pytorch.llm.rope` | medium | RoPE：建表 + 旋转 |
| `pytorch.llm.top_k_top_p_sampling` | medium | 温度 + top-k + nucleus |
| `pytorch.llm.greedy_beam_search` | medium | Greedy + Beam Search |
| `pytorch.llm.swiglu_ffn` | medium | LLaMA 风格 FFN |
| `pytorch.llm.transformer_block` | hard | 完整 LLaMA block（集成题） |

## 判分原理

- **数值对比**：`torch.allclose` 语义，支持 ndarray / Tensor / scalar
- **自定义判定**：支持 shape 检查、init 分布检查等
- **种子固定**：`random` / `numpy` / `torch` 全栈 seed，保证可复现
- **设备自动选择**：cuda → mps → cpu，MPS 自动放宽容差
- **运行时限**：每题 < 2 秒，超时警告

## 项目结构

```
├── mlleetcode/            # 核心框架
│   ├── cli.py             # CLI 入口
│   ├── judge.py           # 判分引擎
│   ├── registry.py        # 题目发现与加载
│   ├── report.py          # Rich 终端输出
│   ├── reference.py       # 集成题的参考工具函数
│   ├── server/            # FastAPI 后端
│   └── utils/             # seed / compare / sandbox / ...
├── problems/              # 题库（每题 5 文件）
│   ├── numpy/{basics,ml}/
│   └── pytorch/{basics,ml,nn,llm}/
├── web/                   # React 前端
│   ├── src/
│   └── dist/              # 构建产物（已提交，用户无需装 node）
├── workspace/             # 用户写代码的地方
├── tests/                 # 框架测试
└── docs/AUTHORING.md      # 出题指南
```

## 贡献 / 加题

参考 [docs/AUTHORING.md](docs/AUTHORING.md)，每道题是 5 个文件：
`meta.yaml` + `README.md` + `starter.py` + `solution.py` + `test_cases.py`。

加完后跑 `mlleetcode verify <id>` 确认 100/100 即可。

## 开发

```bash
pip install -e ".[web]"
pytest -q                     # 框架测试
mlleetcode verify             # 全题自检

# 前端开发（需要 node）
cd web && npm install && npm run dev    # dev server :5173
# 后端单独起
uvicorn mlleetcode.server.app:app --reload --port 8000
```

## License

MIT
