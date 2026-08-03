"""Test cases for pytorch.llm.ppo_clip_loss (KL penalty + GAE + clipped surrogate).

Compares against the reference plus known-value checks:
- kl_coef == 0 and logp arbitrary: reduces to plain-reward GAE PPO.
- dones all 1 and logratio == 0: A_t = r'_t - V(s_t), loss = -mean(r' - V[:T]),
  where r' = rewards - kl_coef * k3.
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


def _run(
    user_module,
    logratio,
    logp,
    logp_ref,
    rewards,
    values,
    dones,
    gamma,
    lam,
    eps,
    kl_coef,
):
    args = (logratio, logp, logp_ref, rewards, values, dones)
    return (
        user_module.ppo_clip_loss(*[a.clone() for a in args], gamma, lam, eps, kl_coef),
        _REF.ppo_clip_loss(*[a.clone() for a in args], gamma, lam, eps, kl_coef),
    )


def _check_all_terminal(user_module):
    """dones all 1, logratio 0 → loss = -mean(r' - V[:T]), r' = rewards - kl*k3."""
    T = 6
    logp = _randn(T, 40)
    logp_ref = _randn(T, 41)
    rewards = _randn(T, 42)
    values = _randn(T + 1, 43)
    dones = torch.ones(T)
    logratio = torch.zeros(T)
    kl_coef = 0.1
    out = user_module.ppo_clip_loss(
        logratio.clone(),
        logp.clone(),
        logp_ref.clone(),
        rewards.clone(),
        values.clone(),
        dones.clone(),
        0.99,
        0.95,
        0.2,
        kl_coef,
    )
    logr = logp_ref - logp
    kl = torch.exp(logr) - 1.0 - logr
    r_pen = rewards - kl_coef * kl
    expected = -(r_pen - values[:T]).mean()
    return compare_numeric(out, expected, atol=1e-6, rtol=1e-6)


TEST_CASES = [
    TestCase(
        name="ppo / KL+GAE, small logratio",
        runner=lambda m: _run(
            m,
            _randn(8, 0, 0.05),
            _randn(8, 1),
            _randn(8, 2),
            _randn(8, 3),
            _randn(9, 4),
            torch.zeros(8),
            0.99,
            0.95,
            0.2,
            0.1,
        ),
        weight=3.0,
        atol=1e-6,
        rtol=1e-6,
    ),
    TestCase(
        name="ppo / with terminals, clip triggered",
        runner=lambda m: _run(
            m,
            _randn(10, 5, 1.0),
            _randn(10, 6),
            _randn(10, 7),
            _randn(10, 8),
            _randn(11, 9),
            _dones(10, 10),
            0.95,
            0.9,
            0.2,
            0.2,
        ),
        weight=3.0,
        atol=1e-6,
        rtol=1e-6,
    ),
    TestCase(
        name="ppo / kl_coef=0 (no KL penalty)",
        runner=lambda m: _run(
            m,
            _randn(8, 11, 0.5),
            _randn(8, 12),
            _randn(8, 13),
            _randn(8, 14),
            _randn(9, 15),
            _dones(8, 16),
            1.0,
            1.0,
            0.1,
            0.0,
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
