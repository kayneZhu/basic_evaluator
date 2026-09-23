"""
Config-driven §5.1 benchmark roster.

The set is not frozen and must not be a hardcoded nine. The same
``benchmark_id`` may appear twice under different protocol profiles
(surface / K / temperature). Append entries here; do not special-case
a length of nine in code.

Wired now: AIME24+25+26 union (K=512) and held-out H (K=128), both
through the C.1 mixin. Nine-benchmark and harm/OOD entries are
appended when the lead names them. HumanEval+ is registered on the
factory as a stub (``NotImplementedError``), not as a live roster job.

OR1-200 is retired. Held-out bins come from H (150/bin, |H|=600),
binned by Base pass@16 under the eval sampler — not from
``ttn_test_200.jsonl`` ``pass_count`` / ``pass_rate``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .length import STUDENT_MAX_NEW_TOKENS
from .paths import protocol_profile_name, samples_output_dir
from .sampling import EVAL_TEMPERATURE, EVAL_TOP_P
from .validation import (
    DEFAULT_HELDOUT_H_PATH,
    DT_TRAIN_SUBSET_SIZE,
    VAL_BENCHMARK_DT_TRAIN,
    VAL_BENCHMARK_HELD_OUT,
    VAL_K,
    VAL_SURFACE_DT_TRAIN,
    VAL_SURFACE_HELD_OUT,
    VAL_TEMPERATURE,
    VAL_TOP_P,
)


@dataclass(frozen=True)
class ProtocolProfile:
    surface: str
    benchmark_id: str
    k: int
    temperature: float
    adaptor_key: str
    data_path: str
    top_p: float = EVAL_TOP_P
    max_new_tokens: int = STUDENT_MAX_NEW_TOKENS
    notes: str = ""

    @property
    def profile_name(self) -> str:
        return protocol_profile_name(self.surface, self.k, self.temperature)

    def output_dir(self, outputs_root, run_id, student_slug, step):
        return samples_output_dir(
            outputs_root,
            run_id,
            student_slug,
            step,
            self.benchmark_id,
            self.surface,
            self.k,
            self.temperature,
        )


# Live paper jobs. This is a list, not a closed set of nine.
ROSTER: List[ProtocolProfile] = [
    ProtocolProfile(
        surface="aime_union",
        benchmark_id="aime_union",
        k=512,
        temperature=EVAL_TEMPERATURE,
        top_p=EVAL_TOP_P,
        adaptor_key="c1_aime_union",
        data_path="data/aime24_25_26_bench_schema.jsonl",
        notes=(
            "90-problem AIME24+25+26 union. avg@8 is sample_idx 0–7 of this "
            "same pool. AIME26 is a problem_id prefix filter (aime26:), "
            "not a second draw. Sampler T=0.6 / top_p=0.95."
        ),
    ),
    ProtocolProfile(
        surface="heldout_h",
        benchmark_id=VAL_BENCHMARK_HELD_OUT,
        k=128,
        temperature=EVAL_TEMPERATURE,
        top_p=EVAL_TOP_P,
        adaptor_key="c1_or1_200",
        data_path=str(DEFAULT_HELDOUT_H_PATH),
        notes=(
            "Held-out H (|H|=600, 150/bin). Supply Base pass@16 bin "
            "labels from the training side; OR1-200 is retired."
        ),
    ),
    # Cheap mid-run validation (INTERFACE.md §5). Same C.1 adaptor; K=8.
    # Selection uses val_heldout_h only; val_dt_train is for curves.
    ProtocolProfile(
        surface=VAL_SURFACE_HELD_OUT,
        benchmark_id=VAL_BENCHMARK_HELD_OUT,
        k=VAL_K,
        temperature=VAL_TEMPERATURE,
        top_p=VAL_TOP_P,
        adaptor_key="c1_or1_200",
        data_path=str(DEFAULT_HELDOUT_H_PATH),
        notes=(
            "Mid-run validation on held-out H every 50 steps. "
            "pass@1 (sample_idx 0) + pass@8 from the same K=8 pool. "
            "Always g=0 / τ=0. Primary metric for best-ckpt selection. "
            "Sampler T=0.6 / top_p=0.95."
        ),
    ),
    ProtocolProfile(
        surface=VAL_SURFACE_DT_TRAIN,
        benchmark_id=VAL_BENCHMARK_DT_TRAIN,
        k=VAL_K,
        temperature=VAL_TEMPERATURE,
        top_p=VAL_TOP_P,
        adaptor_key="c1_or1_200",
        data_path="data/dt_train_subset.jsonl",
        notes=(
            f"Mid-run train-curve surface: seeded Dt subset "
            f"(default n={DT_TRAIN_SUBSET_SIZE}). Same pass@1/pass@8 "
            "estimators as held-out. Never used for checkpoint selection. "
            "Training supplies the subset jsonl path."
        ),
    ),
]


def roster_by_surface(surface: str) -> List[ProtocolProfile]:
    return [p for p in ROSTER if p.surface == surface]


def find_profile(
    *,
    surface: str,
    benchmark_id: str,
    k: Optional[int] = None,
) -> ProtocolProfile:
    hits = [
        p for p in ROSTER
        if p.surface == surface and p.benchmark_id == benchmark_id
        and (k is None or p.k == k)
    ]
    if len(hits) != 1:
        raise KeyError(
            f"expected one roster entry for surface={surface!r} "
            f"benchmark_id={benchmark_id!r} k={k!r}, got {len(hits)}"
        )
    return hits[0]
