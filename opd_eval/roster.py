"""
Config-driven §5.1 benchmark roster.

The set is not frozen and must not be a hardcoded nine. The same
``benchmark_id`` may appear twice under different protocol profiles
(surface / K / temperature). Append entries here; do not special-case
a length of nine in code.

Wired now: AIME24+25+26 union (K=512) and OR1-200 held-out (K=128),
both through the C.1 mixin. Nine-benchmark, harm/OOD, and coding
entries are added when the lead names them.

Bins for OR1-200 are a training-side input (fresh Base pass@16). They
are not a roster field and must not be read from ``ttn_test_200.jsonl``
``pass_count`` / ``pass_rate`` (previous paper's pass@32).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .paths import protocol_profile_name, samples_output_dir


@dataclass(frozen=True)
class ProtocolProfile:
    surface: str
    benchmark_id: str
    k: int
    temperature: float
    adaptor_key: str
    data_path: str
    max_new_tokens: int = 8192
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
        temperature=1.0,
        adaptor_key="c1_aime_union",
        data_path="data/aime24_25_26_bench_schema.jsonl",
        notes=(
            "90-problem AIME24+25+26 union. avg@8 is sample_idx 0–7 of this "
            "same pool. AIME26 is a problem_id prefix filter (aime26:), "
            "not a second draw."
        ),
    ),
    ProtocolProfile(
        surface="or1_200",
        benchmark_id="or1_200",
        k=128,
        temperature=1.0,
        adaptor_key="c1_or1_200",
        data_path="data/ttn_test_200.jsonl",
        notes=(
            "Held-out bins. Supply Base pass@16 bin assignment from the "
            "training side; do not reuse jsonl pass_count/pass_rate."
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
