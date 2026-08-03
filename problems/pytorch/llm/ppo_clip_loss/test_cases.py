"""Test cases for pytorch.llm.ppo_clip_loss (GAE + clipped surrogate).

Compares against the reference plus known-value checks:
- dones all 1 (every step terminal): GAE degenerates to A_t = r_t - V(s_t),
  so with logratio == 0 the loss = -mean(r - V[:T]).
- logratio == 0 (ratio == 1): loss = -mean(advantages).
"""

from __future__ import annotations

from pathlib import Path

import torch

from mlleetcode.judge import TestCase
from mlleetcode.utils.compare import compare_numeric
from mlleetcode.utils.sandbox import load_module_from_path

_REF = load_module_from_path(Path(__file__).with_name("solution.py"), "ref_ppo")


def _g(seed: int):
    return torch.Generator().manual_seed(seed)


def _randn(n, seed, scale=1.0):
    return torch.randn(n, generator=_g(seed)) * scale


def _dones(T, seed, p=0.2):
    return (torch.rand(T, generator=_g(seed)) < p).float()


def _run(user_module, logratio, rewards, values, dones, gamma, lam, eps):
    args = (logratio, rewards, values, dones)
    return (
        user_module.ppo_clip_loss(*[a.clone() for a in args], gamma, lam, eps),
        _REF.ppo_clip_loss(*[a.clone() for a in args], gamma, lam, eps),
    )


def _check_all_terminal(user_module):
    """dones all 1 → A_t = r_t - V(s_t); logratio 0 → loss = -mean(r - V[:T])."""
    T = 6
    rewards = _randn(T, 40)
    values = _randn(T + 1, 41)
    dones = torch.ones(T)
    logratio = torch.zeros(T)
    out = user_module.ppo_clip_loss(
        logratio.clone(),
        rewards.clone(),
        values.clone(),
        dones.clone(),
        0.99,
        0.95,
        0.2,
    )
    expected = -(rewards - values[:T]).mean()
    return compare_numeric(out, expected, atol=1e-6, rtol=1e-6)


TEST_CASES = [
    TestCase(
        name="ppo / GAE no terminals, small logratio",
        runner=lambda m: _run(
            m,
            _randn(8, 0, 0.05),
            _randn(8, 1),
            _randn(9, 2),
            torch.zeros(8),
            0.99,
            0.95,
            0.2,
        ),
        weight=3.0,
        atol=1e-6,
        rtol=1e-6,
    ),
    TestCase(
        name="ppo / GAE with terminals, clip triggered",
        runner=lambda m: _run(
            m,
            _randn(10, 3, 1.0),
            _randn(10, 4),
            _randn(11, 5),
            _dones(10, 6),
            0.95,
            0.9,
            0.2,
        ),
        weight=3.0,
        atol=1e-6,
        rtol=1e-6,
    ),
    TestCase(
        name="ppo / gamma=1, lam=1 (Monte-Carlo advantage)",
        runner=lambda m: _run(
            m,
            _randn(8, 7, 0.5),
            _randn(8, 8),
            _randn(9, 9),
            _dones(8, 10),
            1.0,
            1.0,
            0.1,
        ),
        weight=2.0,
        atol=1e-6,
        rtol=1e-6,
    ),
    TestCase(
        name="ppo / all-terminal degenerate check",
        runner=_check_all_terminal,
        weight=2.0,
    ),
]
