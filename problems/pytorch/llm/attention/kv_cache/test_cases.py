"""Test cases for pytorch.llm.attention.kv_cache."""

from __future__ import annotations

from math import sqrt
from pathlib import Path

import torch
import torch.nn.functional as F

from mlleetcode.judge import TestCase
from mlleetcode.utils.compare import CompareResult, compare_numeric
from mlleetcode.utils.sandbox import load_module_from_path

_REF = load_module_from_path(Path(__file__).with_name("solution.py"), "ref_kvcache")


def _g(seed):
    return torch.Generator().manual_seed(seed)


def _full_sdpa(q, k, v):
    """单步无 cache 的参考 SDPA。"""
    d = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / sqrt(d)
    attn = F.softmax(scores, dim=-1)
    return attn @ v


def _run_basic(user_module, B=2, H=2, T_past=3, T_new=1, D=8, seed=0):
    g = _g(seed)
    q_new = torch.randn(B, H, T_new, D, generator=g)
    k_new = torch.randn(B, H, T_new, D, generator=g)
    v_new = torch.randn(B, H, T_new, D, generator=g)
    k_cache = torch.randn(B, H, T_past, D, generator=g)
    v_cache = torch.randn(B, H, T_past, D, generator=g)

    out, new_k, new_v = user_module.sdpa_with_kv_cache(
        q_new.clone(),
        k_new.clone(),
        v_new.clone(),
        k_cache.clone(),
        v_cache.clone(),
    )
    ref_out, ref_k, ref_v = _REF.sdpa_with_kv_cache(
        q_new.clone(),
        k_new.clone(),
        v_new.clone(),
        k_cache.clone(),
        v_cache.clone(),
    )
    # 把三个输出拼到一起做单次 numeric 对比
    return (
        torch.cat([out.reshape(-1), new_k.reshape(-1), new_v.reshape(-1)]),
        torch.cat([ref_out.reshape(-1), ref_k.reshape(-1), ref_v.reshape(-1)]),
    )


def _run_first_step_no_cache(user_module):
    """首步：k_cache / v_cache 为 None。输出应等同于不带 cache 的 SDPA。"""
    B, H, T, D = 1, 2, 4, 8
    g = _g(99)
    q_new = torch.randn(B, H, T, D, generator=g)
    k_new = torch.randn(B, H, T, D, generator=g)
    v_new = torch.randn(B, H, T, D, generator=g)
    out, k, v = user_module.sdpa_with_kv_cache(
        q_new.clone(),
        k_new.clone(),
        v_new.clone(),
        None,
        None,
    )
    expected = _full_sdpa(q_new, k_new, v_new)
    return out, expected


def _run_equivalence_to_prefill(user_module) -> CompareResult:
    """核心 property：T 步增量 == 一次性 prefill。"""
    B, H, T_full, D = 1, 2, 5, 8
    g = _g(7)
    full_k = torch.randn(B, H, T_full, D, generator=g)
    full_v = torch.randn(B, H, T_full, D, generator=g)
    last_q = torch.randn(B, H, 1, D, generator=g)

    # 一次性 prefill 的最后一步输出
    expected = _full_sdpa(last_q, full_k, full_v)

    # 增量：第 0..T_full-2 步把 K/V 累积到 cache（这些步的 q 用占位）
    k_cache = full_k[:, :, : T_full - 1].clone()
    v_cache = full_v[:, :, : T_full - 1].clone()

    # 最后一步用 last_q + 第 T_full-1 个 k/v
    out, _, _ = user_module.sdpa_with_kv_cache(
        last_q,
        full_k[:, :, T_full - 1 : T_full].clone(),
        full_v[:, :, T_full - 1 : T_full].clone(),
        k_cache,
        v_cache,
    )
    return compare_numeric(out, expected, atol=1e-5, rtol=1e-5)


TEST_CASES = [
    TestCase(
        name="single-step append / values + cache",
        runner=lambda m: _run_basic(m, seed=1),
        weight=2.0,
        atol=1e-5,
        rtol=1e-5,
    ),
    TestCase(
        name="multi-batch / values + cache",
        runner=lambda m: _run_basic(m, B=3, H=4, T_past=5, T_new=2, D=16, seed=2),
        weight=2.0,
        atol=1e-5,
        rtol=1e-5,
    ),
    TestCase(
        name="first step (k_cache=None)",
        runner=_run_first_step_no_cache,
        weight=2.0,
        atol=1e-5,
        rtol=1e-5,
    ),
    TestCase(
        name="property / incremental ≡ prefill",
        runner=_run_equivalence_to_prefill,
        weight=3.0,
    ),
]
