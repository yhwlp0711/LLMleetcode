# Authoring New Problems

Each problem is a directory under `problems/<framework>/<category>/<slug>/`
containing five files. The framework and category are arbitrary nesting; common
ones today are:

```
problems/numpy/basics/...
problems/numpy/ml/...
problems/pytorch/basics/...
problems/pytorch/ml/...
problems/pytorch/nn/...
problems/pytorch/llm/...
```

## Required files

```
problems/<framework>/<category>/<slug>/
├── meta.yaml         # title, difficulty, framework, tags, timeout
├── README.md         # problem statement (rendered by `mlleetcode show`)
├── starter.py        # scaffold given to the user; raises NotImplementedError
├── solution.py       # reference impl (drives expected outputs & `verify`)
└── test_cases.py     # defines TEST_CASES: list[TestCase]
```

`meta.yaml` example:

```yaml
title: "Scaled Dot-Product Attention"
difficulty: easy           # easy | medium | hard
framework: pytorch         # numpy | pytorch | mixed
tags: [attention, transformer, llm]
timeout: 10                # seconds per test case
entrypoint: sdpa           # advisory only — what function the user implements
```

## Two problem patterns

### Pattern A: Operator (recommended default)

The user implements a **pure function** taking explicit inputs (including any
weights). The judge constructs the inputs deterministically and feeds the same
tensors to both the user and the reference. **Zero randomness in the user's
code.**

Example: `pytorch.llm.attention.scaled_dot_product_attention` —
`sdpa(q, k, v, mask) -> output`.

This is the cleanest pattern — no init-order ambiguity, no parameter-naming
conventions to enforce, judging is straight numeric comparison.

### Pattern B: nn.Module

The user implements a `class XXX(nn.Module)` with `__init__` (containing
parameter creation/initialization) and `forward`. Used when the *coding
exercise* explicitly includes parameter initialization, weight naming, etc.

Split judging into two kinds of cases:

1. **Init tests** — check parameter shapes, names, and **distribution
   statistics** (mean/std) using `mlleetcode.utils.stats`. Exact values cannot
   be compared because `nn.Module` init order matters.
2. **Forward tests** — first call
   `mlleetcode.utils.weights.sync_weights(user_module, ref_module)` to copy
   reference weights into the user's module, then compare `user(x)` vs
   `ref(x)` numerically.

This isolates "did you initialize correctly?" from "is your forward correct?".

## Writing test cases

`TEST_CASES` is a list of `TestCase` instances. Each case has a `runner` that
receives the user module and returns one of three forms:

```python
from mlleetcode.judge import TestCase
from mlleetcode.utils.compare import CompareResult

# Form 1: (actual, expected) — engine runs numeric comparison
TestCase(name="forward", runner=lambda m: (m.f(x), ref.f(x)))

# Form 2: CompareResult — runner did its own judging
TestCase(name="param shapes", runner=lambda m: check_param_shapes(m))

# Form 3: bool — pass/fail with no detail (use sparingly)
TestCase(name="smoke", runner=lambda m: m.thing() is not None)
```

Common case attributes:

| Field | Default | Notes |
|---|---|---|
| `name` | required | Shown in the report. Use `"category / sub-name"` to group. |
| `runner` | required | Callable returning one of the three forms above. |
| `weight` | `1.0` | Contribution to the final 0–100 score. |
| `atol` | `1e-5` | Absolute tolerance for numeric form. |
| `rtol` | `1e-4` | Relative tolerance for numeric form. |

### Loading the reference solution

Use `mlleetcode.utils.sandbox.load_module_from_path` (don't `import` the
solution as a normal module — it lives outside the package):

```python
from pathlib import Path
from mlleetcode.utils.sandbox import load_module_from_path

_REF = load_module_from_path(Path(__file__).with_name("solution.py"), "ref_my_problem")
```

### Reusing fixtures

Define `_fx_*()` helpers that build the inputs **deterministically** (use
explicit `torch.Generator(...).manual_seed(...)` rather than the global RNG, so
the inputs are stable regardless of any user-side seeding). Call the helper
inside the lambda so each case gets a fresh tensor (avoids in-place mutation
bugs):

```python
def _fx_small():
    return torch.randn(4, 6, generator=torch.Generator().manual_seed(0))

TEST_CASES = [
    TestCase(name="small", runner=lambda m: (m.f(_fx_small()), _REF.f(_fx_small()))),
]
```

## Verifying your new problem

```bash
mlleetcode verify <slug>           # must print ACCEPTED 100.0/100
pytest -q                          # framework tests must still pass
```

The `verify` command runs the reference `solution.py` through the same judge a
user would hit; if it doesn't pass, your test cases are inconsistent with your
own reference.

## Style tips

- Prefer many small focused cases over one mega-case — when something fails the
  user immediately sees which property broke.
- Use case names like `"forward / no mask"`, `"forward / causal mask"`,
  `"backward / grad w.r.t. x"` to group visually.
- Set `atol`/`rtol` based on expected accumulation error
  (`1e-5/1e-4` for fp32 attention is usually fine; tighten to `1e-6/1e-6` when
  everything is double).
- For problems that aren't deterministic (e.g. KMeans with random init), prefer
  Pattern B with init split out, or have the user accept a `seed` parameter.

## Test runtime budget (hard rule)

`mlleetcode verify <problem>` **must finish in under 2 seconds** in total
across all of that problem's test cases on CPU. The framework prints:

- a [yellow] warning at > 2s
- a [bold red] warning at > 10s (treat as a failed problem; shrink fixtures
  before merging)

### Why such a tight budget?

This judge is meant to give instant feedback like LeetCode. If users wait
> 30s per submission, they stop using it. The budget also forces problem
authors to think about "what is the smallest fixture that still proves
correctness?", which usually produces cleaner, more focused test cases.

### How to stay within budget

The trick is **"prove the algorithm by the smallest possible example"**,
not "run a realistic-sized scenario". Some patterns:

- **Training loops**: pick small `epochs` (≤ 1000) and small data
  (`N ≤ 500`); the reference solution is the oracle, so absolute
  convergence quality doesn't matter — only that user and reference agree
  step-by-step.
- **KMeans / iterative**: cap `max_iter` to 30, use `K ≤ 5`, `N ≤ 500`.
- **Attention**: `B ≤ 2`, `H ≤ 8`, `T ≤ 32`, `D ≤ 64`.
- **Sampling / decoding**: tiny vocab (≤ 50), short sequences
  (`max_len ≤ 16`), small beam (`≤ 4`).
- **KV cache**: don't actually generate 50 tokens; prove "incremental
  append == single-shot prefill" with `T_full ≤ 5`.
- **Beam search / greedy**: abstract the model as a deterministic
  `model_fn(input_ids) -> logits` lookup; don't load an actual LM.

If a test case genuinely needs to be slow (e.g. compares against a
PyTorch op that has setup overhead), keep it as a single case and
mark `timeout` in `meta.yaml` accordingly — but always question whether
a smaller fixture would prove the same property.

## Pitfalls to avoid in problem design

### Dropout

Although the framework pins seeds before every test case, **do not include
`nn.Dropout` (or any RNG-consuming op) in the user-facing surface** unless it's
the actual subject of the problem.

Why: even with a fixed seed, the dropout mask depends on the *order* in which
RNG is consumed. If the user puts dropout after `linear` instead of before, or
adds an init that consumes RNG, the masks diverge and the test fails for a
reason unrelated to the algorithm.

Workarounds when the real-world layer does use dropout (e.g. a Transformer
block):

1. **Force eval mode in the runner**: `user_mod.eval()` and `ref_mod.eval()`
   before the forward — this turns `nn.Dropout` into identity. Note this in
   the README so the user is not surprised.
2. **Make `p=0` the default** in the problem signature, and only test with
   `p=0`.
3. **Pass the dropout mask as an input** if dropout itself is the point of
   the exercise.

### Other RNG-consuming ops

The same caveat applies to `torch.bernoulli`, `torch.multinomial`,
`torch.randn` called inside `forward`, and any custom `nn.init.*` that runs at
forward time. Either ban them in the README or absorb them into the test
fixtures.

### In-place ops on shared inputs

Always clone tensors before passing them to the user's function if the same
fixture is reused by the reference. Otherwise an in-place op (`.relu_()`,
`+=`, `.zero_()`) silently corrupts the expected value:

```python
# BAD: user's in-place op mutates the fixture
x = _fx()
return user.f(x), ref.f(x)

# GOOD: separate copies
x = _fx()
return user.f(x.clone()), ref.f(x.clone())
```
