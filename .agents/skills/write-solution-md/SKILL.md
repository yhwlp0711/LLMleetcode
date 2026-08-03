---
name: write-solution-md
description: 为 mlleetcode 题库编写或改写易懂的题目解析 solution.md。当需要新增题目解析、批量改写解析文案、或统一解析风格时使用。
tags: [mlleetcode, docs, content]
version: "1.0.0"
---

# 编写 mlleetcode 题目解析（solution.md）

为 `problems/**/solution.md` 编写「学习者友好」的解析。目标读者是刚做完题、
想搞懂「怎么想到的、为什么对」的学习者，**不是**在背面试题的人。

## 核心原则

1. **易懂优先**：像讲给同学听，先给直觉再给公式，别只堆数学。
2. **讲「为什么」**：不直接甩答案，要说明解法是怎么推导/想到的。
3. **术语中英对照**：专业术语首次出现给中英对照，例如
   「数值稳定（numerical stability）」「神经元死亡（dead neuron）」
   「梯度消失（vanishing gradient）」「广播（broadcasting）」
   「正则化（regularization）」「奖励模型（reward model）」。
4. **不写「易错点/陷阱」独立模块**（因人而异，价值低）。
5. **延伸不提「面试怎么考」**（面试变化快、无意义）；延伸只讲与其它
   概念/题目的联系。

## 固定 4 模块结构

每篇 solution.md 用以下结构（标题用中文）：

```markdown
# 解题思路：<题目名>

## 一句话思路

<30 秒能抓住要点的 TL;DR：这题核心就是做 X，难点在 Y。>

## 从直觉到公式  （数学题用此标题；算法题用「拆解思路」）

<讲解法是怎么想到的：数学题给关键推导，算法题给分步骤。
重点是「为什么这样做」，而不是直接给结论。>

## 参考实现

​```python
<完整可运行代码，关键行加简短行内注释>
​```

## 关键点

1. **<要点标题>**：<为什么这么做，用直觉解释>
2. ...
N. **延伸**：<与其它概念/题目的联系；用 `pytorch.llm.xxx` 形式引用兄弟题；不提面试>
```

## 写作规范

- 语言口语化、亲切，多用「我们希望…」「关键观察是…」「巧思在于…」。
- 数学：先给直觉再给式子；行间公式用 `$$ ... $$`（**必须独占一行**，
  KaTeX/remark-math 不支持跨行 `$$` 块，否则渲染失败）。
- 行内数学用单 `$...$`，行内代码/API 用反引号（前端会给已知 API 加悬浮提示）。
- 引用兄弟题用完整题目 id 反引号包裹，如 `pytorch.llm.loss.ppo_clip_loss`
  （前端会自动渲染成可点击链接）。id 必须是当前题库里真实存在的。
- 代码要和该题的 `solution.py` 逻辑一致（可精简展示，但不能有错）。
- 关键点里如果涉及数值稳定、维度/广播、API 选择，务必解释「为什么」。

## 术语对照参考（首次出现时给中英）

- 数值稳定 numerical stability / 溢出 overflow / 下溢 underflow
- 广播 broadcasting / 归约 reduce / 维度 dimension
- 平移不变性 shift invariance / 正则化 regularization
- 梯度消失 vanishing gradient / 梯度爆炸 exploding gradient / 神经元死亡 dead neuron
- 奖励模型 reward model / 价值网络 value network / 优势 advantage
- 因果掩码 causal mask / 位置编码 positional encoding

## 参考样板（已按本标准改写，风格对齐它们）

- `problems/pytorch/nn/numeric_activations/solution.md` — 数值稳定 + 术语对照的典型
- `problems/pytorch/llm/loss/dpo_loss/solution.md` — 有数学推导的典型

## 流程

1. 先读该题的 `README.md`（题面/约定）和 `solution.py`（参考实现），确保
   解析与判分口径、公式约定一致。
2. 按 4 模块结构写 `solution.md`。
3. 自查：`$$` 是否独占行；引用的兄弟题 id 是否真实存在；术语是否给了中英对照；
   有没有出现「面试」字样（应删除）；有没有写「易错点」独立模块（应删除）。
4. 不改动 `README.md` / `solution.py` / `test_cases.py` / `meta.yaml`，
   只写 `solution.md`。
