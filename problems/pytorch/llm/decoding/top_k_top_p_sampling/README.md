# Top-k / Top-p (Nucleus) Sampling

实现 LLM 推理时常用的解码策略 —— **温度缩放 + top-k + top-p 过滤**。这是
一道纯函数题，**不做实际采样**（采样涉及随机数，难判分），只做「把不该
采样到的 token 屏蔽掉」。

## 函数签名

```python
def filter_logits(
    logits: torch.Tensor,    # (B, V)  原始 logits
    *,
    temperature: float = 1.0,
    top_k: int = 0,          # 0 表示不启用
    top_p: float = 1.0,      # 1.0 表示不启用
) -> torch.Tensor:           # (B, V)  过滤后的 logits（被屏蔽的位置 = -inf）
```

## 算法

按顺序应用以下过滤：

### 1. 温度缩放

`logits = logits / temperature`

`temperature > 1` 让分布更平（更随机），`< 1` 让分布更尖（更确定）。
`temperature = 1.0` 时不变。

### 2. Top-k 过滤

如果 `top_k > 0`：每行只保留前 `k` 大的 logit，其余置 `-inf`。

并列时实现依赖，本题用 `torch.topk` 的默认顺序即可。

### 3. Top-p (Nucleus) 过滤

如果 `top_p < 1.0`：

1. 把每行 logits 按降序排序，转成概率（softmax）。
2. 计算累积概率。
3. 找出**累积概率首次超过 top_p 的位置**，保留到这里为止的所有 token。
4. 把其余 token 的 logit 置 `-inf`。

**注意**：要保证至少保留 1 个 token（即使最大概率 > top_p）。

## 说明

- **不调用任何**库的 sampling/filter 函数（如 `torch.multinomial`、
  HuggingFace 的 `TopKLogitsWarper`）—— 自己写。
- 返回**过滤后的 logits**（保留位置 = 原 logits / temperature；屏蔽位置
  = `-inf`），让下游随便用 softmax + sample。
- 三个过滤组合时按上述顺序：temperature → top-k → top-p。
