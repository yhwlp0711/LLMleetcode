// 常见 NumPy / PyTorch API 的简明说明，用于题面/解析里 inline code 的悬浮提示。
// key 用 API 的规范写法（去掉参数），匹配时会做归一化（见 apiTooltip.tsx）。

export interface ApiDoc {
  sig: string // 简化签名
  desc: string // 一句话作用
  op: string // 具体做了什么操作
  inputs: string // 输入
  outputs: string // 输出
}

export const API_DOCS: Record<string, ApiDoc> = {
  // ---- NumPy ----
  'np.where': {
    sig: 'np.where(cond, a, b)',
    desc: '向量化的三元运算符：逐元素在 a、b 之间选一个。',
    op: '遍历每个位置，若该位置 cond 为 True 就取 a 的对应元素，否则取 b 的对应元素，等价于 `a if cond else b` 的逐元素版本。',
    inputs: 'cond 布尔数组；a、b 与 cond 可广播。',
    outputs: 'cond 为 True 处取 a，否则取 b，形状同广播结果。',
  },
  'np.argmax': {
    sig: 'np.argmax(x, axis=None)',
    desc: '返回沿指定轴最大值所在的索引。',
    op: '沿 axis 扫描，找出最大值的位置下标；多个并列时返回第一个（最小索引）。',
    inputs: 'x 数组；axis 指定轴（None 表示展平）。',
    outputs: '最大值的索引（并列时取最小索引）。',
  },
  'np.moveaxis': {
    sig: 'np.moveaxis(x, src, dst)',
    desc: '把某个轴移动到新位置，比 transpose 更易读。',
    op: '把第 src 个轴抽出来插到第 dst 位，其余轴顺序不变；只改变维度排列，不复制数据。',
    inputs: 'x 数组；src/dst 原位置与目标位置。',
    outputs: '轴被重排后的视图（不复制数据）。',
  },
  'np.cumsum': {
    sig: 'np.cumsum(x, axis)',
    desc: '沿轴做累加（前缀和）。',
    op: '沿 axis 从头往后逐个累加：结果第 i 个元素 = 输入前 i 个元素之和。',
    inputs: 'x 数组；axis 累加的轴。',
    outputs: '同形数组，每个位置是该轴上到此为止的累加和。',
  },
  'np.bincount': {
    sig: 'np.bincount(x, weights, minlength)',
    desc: '统计非负整数数组里每个值出现的次数（或加权和）。',
    op: '把每个整数当作桶的下标，累加计数（有 weights 时累加权重），相当于向量化的直方图。',
    inputs: 'x 非负整数一维数组；weights 可选权重。',
    outputs: '长度为 max(x)+1 的计数数组。',
  },
  'np.concatenate': {
    sig: 'np.concatenate([a, b, ...], axis)',
    desc: '沿已有轴把多个数组拼接起来。',
    op: '沿 axis 把多个数组首尾相接，该轴长度相加，其余维度必须一致。',
    inputs: '数组列表；axis 拼接的轴（其余维度须一致）。',
    outputs: '拼接后的数组。',
  },
  'np.einsum': {
    sig: 'np.einsum("ij,jk->ik", a, b)',
    desc: '爱因斯坦求和：用下标记法表达乘加、转置、求和等。',
    op: '按下标字符串：重复出现的下标做乘法并求和，未出现在输出的下标被求和掉，可一式表达矩阵乘/转置/求和。',
    inputs: '下标字符串 + 若干数组。',
    outputs: '按下标规则收缩后的数组。',
  },
  'np.diff': {
    sig: 'np.diff(x, axis)',
    desc: '沿轴求相邻元素的差分。',
    op: '沿 axis 计算 x[i+1] - x[i]，输出比输入在该轴上少一个元素。',
    inputs: 'x 数组；axis 差分的轴。',
    outputs: '该轴长度减 1 的差分数组。',
  },
  'np.linalg.svd': {
    sig: 'np.linalg.svd(A)',
    desc: '奇异值分解 A = U·Σ·Vᵀ。',
    op: '把矩阵分解为两个正交矩阵 U、V 和一组非负奇异值，用于 PCA、降维、求秩等。',
    inputs: 'A 二维矩阵。',
    outputs: 'U、奇异值 s、Vᵀ 三部分。',
  },

  // ---- PyTorch ----
  'torch.where': {
    sig: 'torch.where(cond, a, b)',
    desc: '向量化三元运算符：逐元素在 a、b 之间选。',
    op: '逐元素判断 cond：True 取 a 的对应元素，False 取 b 的对应元素。',
    inputs: 'cond 布尔张量；a、b 可广播。',
    outputs: 'cond 为 True 取 a，否则取 b。',
  },
  'torch.cat': {
    sig: 'torch.cat([a, b, ...], dim)',
    desc: '沿已有维度拼接张量。',
    op: '沿 dim 把多个张量首尾相接，该维长度相加，其余维度须一致（不新增维度）。',
    inputs: '张量列表；dim 拼接的维度。',
    outputs: '拼接后的张量。',
  },
  'torch.stack': {
    sig: 'torch.stack([a, b, ...], dim)',
    desc: '在新维度上堆叠多个同形张量。',
    op: '在 dim 处插入一个新维度，把 N 个同形张量叠成一层，结果比输入多一维。',
    inputs: '同形张量列表；dim 新维度位置。',
    outputs: '比输入多一维的张量。',
  },
  'torch.arange': {
    sig: 'torch.arange(start, end, step)',
    desc: '生成等差整数/浮点序列（不含 end）。',
    op: '从 start 开始每次加 step，直到（不含）end，生成一维等差序列。',
    inputs: 'start、end、step。',
    outputs: '一维张量。',
  },
  'torch.exp': {
    sig: 'torch.exp(x)',
    desc: '逐元素求自然指数 eˣ。',
    op: '对每个元素计算 e 的该值次幂。',
    inputs: 'x 张量。',
    outputs: '同形张量。',
  },
  'torch.log': {
    sig: 'torch.log(x)',
    desc: '逐元素求自然对数 ln(x)。',
    op: '对每个元素取自然对数；输入需为正，否则得到 -inf 或 nan。',
    inputs: 'x 正值张量。',
    outputs: '同形张量。',
  },
  'torch.logsumexp': {
    sig: 'torch.logsumexp(x, dim)',
    desc: '数值稳定地计算 log(Σ eˣ)，softmax/交叉熵的核心。',
    op: '先减去该维最大值再 exp、求和、取 log、加回最大值，避免 exp 溢出，等价于 log(Σ eˣ)。',
    inputs: 'x 张量；dim 归约的维度。',
    outputs: '沿 dim 归约后的张量。',
  },
  'torch.sigmoid': {
    sig: 'torch.sigmoid(x)',
    desc: '逐元素 sigmoid：1/(1+e⁻ˣ)，映射到 (0,1)。',
    op: '对每个元素计算 1/(1+e⁻ˣ)，把任意实数压到 (0,1)，常用作二分类概率。',
    inputs: 'x 张量。',
    outputs: '同形张量，值域 (0,1)。',
  },
  'torch.softmax': {
    sig: 'torch.softmax(x, dim)',
    desc: '沿维度归一化成概率分布（和为 1）。',
    op: '沿 dim 对每个元素求 eˣ 再除以该维的 eˣ 之和，得到非负且和为 1 的概率。',
    inputs: 'x 张量；dim 归一化的维度。',
    outputs: '同形张量，沿 dim 求和为 1。',
  },
  'F.softmax': {
    sig: 'F.softmax(x, dim)',
    desc: '沿维度归一化成概率分布（和为 1）。',
    op: '沿 dim 对每个元素求 eˣ 再除以该维的 eˣ 之和，得到非负且和为 1 的概率。',
    inputs: 'x 张量；dim 归一化的维度。',
    outputs: '同形张量，沿 dim 求和为 1。',
  },
  'F.log_softmax': {
    sig: 'F.log_softmax(x, dim)',
    desc: '数值稳定的 log(softmax(x))。',
    op: '等价于先 softmax 再取 log，但内部用 x - logsumexp(x) 直接算，避免中间概率下溢。',
    inputs: 'x 张量；dim 维度。',
    outputs: '同形张量（对数概率，≤ 0）。',
  },
  'F.logsigmoid': {
    sig: 'F.logsigmoid(x)',
    desc: '数值稳定的 log(sigmoid(x))，DPO 等损失常用。',
    op: '直接算 -softplus(-x) = -log(1+e⁻ˣ)，避免先 sigmoid 再 log 在负值区下溢。',
    inputs: 'x 张量。',
    outputs: '同形张量（≤ 0）。',
  },
  'torch.erf': {
    sig: 'torch.erf(x)',
    desc: '误差函数，精确版 GELU 会用到。',
    op: '逐元素计算高斯误差函数 erf(x)，即标准正态积分的两倍减一，用于精确 GELU。',
    inputs: 'x 张量。',
    outputs: '同形张量，值域 (-1, 1)。',
  },
  'torch.clamp': {
    sig: 'torch.clamp(x, min, max)',
    desc: '把张量元素裁剪到 [min, max] 区间，PPO 裁剪用它。',
    op: '逐元素：小于 min 的置为 min，大于 max 的置为 max，区间内不变。',
    inputs: 'x 张量；min/max 边界。',
    outputs: '同形张量，超界的被截断。',
  },
  'torch.topk': {
    sig: 'torch.topk(x, k, dim)',
    desc: '取沿维度最大的 k 个值及其索引。',
    op: '沿 dim 排序找出最大的 k 个元素，同时返回它们的值和原始位置索引。',
    inputs: 'x 张量；k 个数；dim 维度。',
    outputs: '(values, indices) 两个张量。',
  },
  'torch.multinomial': {
    sig: 'torch.multinomial(p, n)',
    desc: '按概率分布采样索引，top-k/top-p 采样用它。',
    op: '把 p 当作各类别的概率权重，按此分布随机抽 n 个类别下标（有放回或无放回）。',
    inputs: 'p 概率张量（非负）；n 采样个数。',
    outputs: '采样到的索引张量。',
  },
  'torch.tril': {
    sig: 'torch.tril(x)',
    desc: '取矩阵下三角（上三角置 0），构造因果 mask 用。',
    op: '保留主对角线及以下的元素，对角线以上全部置 0；用于让注意力只看到当前及之前的位置。',
    inputs: 'x 二维/批量方阵。',
    outputs: '下三角保留、其余为 0 的张量。',
  },
  'torch.gather': {
    sig: 'torch.gather(x, dim, index)',
    desc: '沿维度按索引取值，交叉熵取目标类 log-prob 用。',
    op: '沿 dim 按 index 给出的下标逐位置取元素，输出形状与 index 相同；常用来取每行目标类的值。',
    inputs: 'x 张量；dim 维度；index 同形索引张量。',
    outputs: '与 index 同形的张量。',
  },
  'torch.no_grad': {
    sig: 'with torch.no_grad():',
    desc: '上下文内关闭梯度追踪，省显存、加速推理。',
    op: '进入该 with 块后，块内所有张量运算都不构建计算图、不记录梯度，退出后恢复。',
    inputs: '无（上下文管理器）。',
    outputs: '块内张量运算不记录计算图。',
  },
  'torch.zeros': {
    sig: 'torch.zeros(*shape)',
    desc: '创建指定形状的全 0 张量。',
    op: '按给定形状分配一块新张量并把所有元素初始化为 0。',
    inputs: 'shape；可选 dtype/device。',
    outputs: '全 0 张量。',
  },
  'torch.ones': {
    sig: 'torch.ones(*shape)',
    desc: '创建指定形状的全 1 张量。',
    op: '按给定形状分配一块新张量并把所有元素初始化为 1。',
    inputs: 'shape；可选 dtype/device。',
    outputs: '全 1 张量。',
  },
}
