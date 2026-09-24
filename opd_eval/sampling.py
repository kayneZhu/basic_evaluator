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
# Mini-protocol / paper ckpt eval base seed (must ≠ binning / C00 / C00′).
EVAL_SEED_MINI_PROTOCOL = 20261001
EVAL_SEEDS: frozenset[int] = frozenset(
    {
        EVAL_SEED_C00,
        EVAL_SEED_C00P,
        EVAL_SEED_BOOTSTRAP_DEFAULT,
        EVAL_SEED_MINI_PROTOCOL,
    }
)

# Pass@16 / H / Dt construction only. Not an evaluation seed.
BINNING_SEED = 20260924

assert BINNING_SEED not in EVAL_SEEDS, "binning seed collides with EVAL_SEEDS"
assert EVAL_SEED_MINI_PROTOCOL != BINNING_SEED
assert EVAL_SEED_MINI_PROTOCOL not in {EVAL_SEED_C00, EVAL_SEED_C00P}


def refuse_binning_seed(seed: int, *, context: str = "eval") -> None:
    """Raise if ``seed`` is the corpus/binning seed (INTERFACE.md §4b)."""
    s = int(seed)
    if s == BINNING_SEED:
        raise ValueError(
            f"{context} seed={s} is BINNING_SEED; use an evaluation seed "
            f"(e.g. EVAL_SEED_MINI_PROTOCOL={EVAL_SEED_MINI_PROTOCOL}) "
            f"outside {{{BINNING_SEED}}}"
        )


def sample_seed(base_seed: int, problem_id: str, sample_idx: int) -> int:
    """Per-request seed. Byte-stable with ``train`` ``rollout_passk.sample_seed``."""
    refuse_binning_seed(base_seed, context="sample_seed")
    key = f"{base_seed}|{problem_id}|passk|{sample_idx}"
    h = 0
    for ch in key:
        h = (h * 131 + ord(ch)) & 0x7FFFFFFF
    return h
