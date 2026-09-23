"""Eval / binning sampler constants (decisions 2026-09-24).

Paper evaluation and pass@16 binning share T=0.6, top_p=0.95.
Training rollouts stay T=1.0, top_p=1.0 — do not import these into train
rollout overrides.

Seed isolation: binning base seeds must be disjoint from every evaluation
seed registered here (see INTERFACE.md).
"""

from __future__ import annotations

# Student math eval (C.1) and mid-run validation.
EVAL_TEMPERATURE = 0.6
EVAL_TOP_P = 0.95

# Registered evaluation seeds (must stay disjoint from binning).
# C00 / C00′ Base noise-floor evals; bootstrap CI default reuses 0.
EVAL_SEED_C00 = 0
EVAL_SEED_C00P = 1
EVAL_SEED_BOOTSTRAP_DEFAULT = 0
EVAL_SEEDS: frozenset[int] = frozenset(
    {
        EVAL_SEED_C00,
        EVAL_SEED_C00P,
        EVAL_SEED_BOOTSTRAP_DEFAULT,
    }
)

# Pass@16 / H / Dt construction only. Not an evaluation seed.
BINNING_SEED = 20260924

assert BINNING_SEED not in EVAL_SEEDS, "binning seed collides with EVAL_SEEDS"
