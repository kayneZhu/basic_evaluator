"""
Cheap τ=0 validation metrics and best/final checkpoint selection.

Training saves weights every ``CKPT_SAVE_CADENCE_STEPS`` (100) and runs
validation every ``VAL_CADENCE_STEPS`` (50). Selection looks only at
**saved** steps that have a held-out validation row, picks the best by
``PRIMARY_METRIC`` (pass@8 on held-out H), and always reports that best
together with the final saved step.

Validation is always g=0 / τ=0 (student solves independently). Sampler
matches paper eval: T=0.6, top_p=0.95. The same K=8 pool yields both
pass@1 (sample_idx 0) and pass@8 (1[k≥1] over 0..7). A Dt train-subset
surface uses the same estimators for train-vs-held-out curves and is
**never** used for selection. OR1-200 is retired.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .length import STUDENT_MAX_NEW_TOKENS
from .sampling import EVAL_TEMPERATURE, EVAL_TOP_P
from .stats import Pool, StatsError, mean_pass_at_k, per_problem_stats


# ── Protocol (INTERFACE.md §5) ───────────────────────────────────────────────

VAL_K = 8
VAL_TEMPERATURE = EVAL_TEMPERATURE
VAL_TOP_P = EVAL_TOP_P
VAL_CADENCE_STEPS = 50
CKPT_SAVE_CADENCE_STEPS = 100

# Surfaces. Selection uses held-out H only (OR1-200 retired).
VAL_SURFACE_HELD_OUT = "val_heldout_h"
VAL_SURFACE_DT_TRAIN = "val_dt_train"
VAL_BENCHMARK_HELD_OUT = "heldout_h"
VAL_BENCHMARK_DT_TRAIN = "dt_train"

# Canonical H path on the GPU box (fields = OR1 pool + ``bin``).
DEFAULT_HELDOUT_H_PATH = Path(
    "/root/autodl-tmp/data/processed/heldout_h600/h600.jsonl"
)

PRIMARY_METRIC = "pass_at_8"
SELECTION_SURFACE = VAL_SURFACE_HELD_OUT

# Default |Dt| subset size for the train-curve surface (seeded subsample).
DT_TRAIN_SUBSET_SIZE = 64


@dataclass(frozen=True)
class ValidationProtocol:
    """Cheap mid-run eval settings. Always τ=0 / g=0."""

    k: int = VAL_K
    temperature: float = VAL_TEMPERATURE
    top_p: float = VAL_TOP_P
    max_new_tokens: int = STUDENT_MAX_NEW_TOKENS
    cadence_steps: int = VAL_CADENCE_STEPS
    ckpt_save_cadence_steps: int = CKPT_SAVE_CADENCE_STEPS
    primary_metric: str = PRIMARY_METRIC
    selection_surface: str = SELECTION_SURFACE
    heldout_h_path: str = str(DEFAULT_HELDOUT_H_PATH)
    g: int = 0  # τ=0; INTERFACE invariant

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_PROTOCOL = ValidationProtocol()


def is_saved_checkpoint_step(
    step: int, *, save_cadence: int = CKPT_SAVE_CADENCE_STEPS
) -> bool:
    if step <= 0:
        return False
    return step % save_cadence == 0


def is_validation_step(
    step: int, *, cadence: int = VAL_CADENCE_STEPS
) -> bool:
    if step <= 0:
        return False
    return step % cadence == 0


def pass_at_1_from_pool(pool: Pool) -> float:
    """
    Mean over problems of ``verified[sample_idx=0]``.

    Uses the first draw of the shared K=8 pool (not a separate K=1 job).
    """
    if not pool:
        raise StatsError("empty sample pool")
    for pid, verified in pool.items():
        if len(verified) < 1:
            raise StatsError(f"{pid}: need sample_idx 0 for pass@1")
    return sum(1.0 if v[0] else 0.0 for v in pool.values()) / len(pool)


def pass_at_8_from_pool(pool: Pool) -> float:
    """Mean over problems of ``1[k ≥ 1]`` on sample_idx 0..7."""
    return mean_pass_at_k(pool, VAL_K)


def metrics_from_pool(pool: Pool, *, k_protocol: int = VAL_K) -> Dict[str, Any]:
    """
    Cheap validation metrics from a complete K-sample pool.

    Requires ``k_protocol ≥ 8`` so pass@8 and (optionally) avg@8 are defined.
    """
    if k_protocol < VAL_K:
        raise StatsError(
            f"validation pool needs K≥{VAL_K}, got {k_protocol}"
        )
    # Truncate view to first VAL_K for the cheap estimators if a larger
    # paper pool is reused; mid-run validation always writes K=8.
    if k_protocol != VAL_K:
        pool = {pid: verified[:VAL_K] for pid, verified in pool.items()}
    stats = per_problem_stats(pool, VAL_K)
    avg8 = sum(s.avg8 for s in stats.values()) / len(stats)  # type: ignore[arg-type]
    return {
        "n_problems": len(pool),
        "k": VAL_K,
        "pass_at_1": pass_at_1_from_pool(pool),
        "pass_at_8": pass_at_8_from_pool(pool),
        "avg_at_8": avg8,
        "g": 0,
        "max_new_tokens": STUDENT_MAX_NEW_TOKENS,
    }


@dataclass(frozen=True)
class StepMetrics:
    step: int
    surface: str
    pass_at_1: float
    pass_at_8: float
    n_problems: int
    benchmark_id: Optional[str] = None
    k: int = VAL_K
    avg_at_8: Optional[float] = None
    g: int = 0
    max_new_tokens: int = STUDENT_MAX_NEW_TOKENS
    extras: Tuple[Tuple[str, Any], ...] = ()

    def metric(self, name: str) -> float:
        if name == "pass_at_1":
            return self.pass_at_1
        if name == "pass_at_8":
            return self.pass_at_8
        if name == "avg_at_8":
            if self.avg_at_8 is None:
                raise KeyError("avg_at_8 missing on this row")
            return self.avg_at_8
        raise KeyError(f"unknown metric {name!r}")

    def as_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "step": self.step,
            "surface": self.surface,
            "benchmark_id": self.benchmark_id,
            "k": self.k,
            "pass_at_1": self.pass_at_1,
            "pass_at_8": self.pass_at_8,
            "n_problems": self.n_problems,
            "g": self.g,
            "max_new_tokens": self.max_new_tokens,
        }
        if self.avg_at_8 is not None:
            d["avg_at_8"] = self.avg_at_8
        for key, value in self.extras:
            d[key] = value
        return d


def parse_metrics_row(raw: Mapping[str, Any]) -> StepMetrics:
    if "step" not in raw:
        raise StatsError("metrics row missing step")
    if "surface" not in raw:
        raise StatsError("metrics row missing surface")
    for key in ("pass_at_1", "pass_at_8", "n_problems"):
        if key not in raw:
            raise StatsError(f"metrics row missing {key}")
    g = int(raw.get("g", 0))
    if g != 0:
        raise StatsError(
            f"validation requires g=0 (τ=0); got g={g} at step={raw['step']}"
        )
    extras = tuple(
        (k, v)
        for k, v in raw.items()
        if k
        not in {
            "step",
            "surface",
            "benchmark_id",
            "k",
            "pass_at_1",
            "pass_at_8",
            "avg_at_8",
            "n_problems",
            "g",
            "max_new_tokens",
        }
    )
    return StepMetrics(
        step=int(raw["step"]),
        surface=str(raw["surface"]),
        benchmark_id=(
            None if raw.get("benchmark_id") is None else str(raw["benchmark_id"])
        ),
        k=int(raw.get("k", VAL_K)),
        pass_at_1=float(raw["pass_at_1"]),
        pass_at_8=float(raw["pass_at_8"]),
        avg_at_8=(
            None if raw.get("avg_at_8") is None else float(raw["avg_at_8"])
        ),
        n_problems=int(raw["n_problems"]),
        g=0,
        max_new_tokens=int(raw.get("max_new_tokens", STUDENT_MAX_NEW_TOKENS)),
        extras=extras,
    )


def load_metrics_jsonl(path: Path) -> List[StepMetrics]:
    rows: List[StepMetrics] = []
    text = Path(path).read_text(encoding="utf-8")
    for line_no, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise StatsError(
                f"{path}:{line_no}: invalid JSON: {exc}"
            ) from exc
        if not isinstance(raw, dict):
            raise StatsError(f"{path}:{line_no}: row must be an object")
        rows.append(parse_metrics_row(raw))
    return rows


def write_metrics_jsonl(path: Path, rows: Iterable[StepMetrics]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r.as_dict(), ensure_ascii=False) for r in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


@dataclass(frozen=True)
class BestFinalReport:
    primary_metric: str
    selection_surface: str
    best: StepMetrics
    final: StepMetrics
    candidates: Tuple[StepMetrics, ...]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "primary_metric": self.primary_metric,
            "selection_surface": self.selection_surface,
            "best": self.best.as_dict(),
            "final": self.final.as_dict(),
            "candidates": [c.as_dict() for c in self.candidates],
            "protocol": DEFAULT_PROTOCOL.as_dict(),
        }


def select_best_and_final(
    rows: Sequence[StepMetrics],
    *,
    saved_steps: Optional[Sequence[int]] = None,
    final_step: Optional[int] = None,
    primary_metric: str = PRIMARY_METRIC,
    selection_surface: str = SELECTION_SURFACE,
    save_cadence: int = CKPT_SAVE_CADENCE_STEPS,
) -> BestFinalReport:
    """
    Pick best among **saved** checkpoints that have a held-out validation row.

    Tie-break: higher ``primary_metric``, then **later** step (prefer the
    more trained of two equal scores). ``final`` is the latest saved step
    that has a row (or ``final_step`` if provided and present).
    """
    held = [r for r in rows if r.surface == selection_surface]
    if not held:
        raise StatsError(
            f"no validation rows for selection_surface={selection_surface!r}"
        )

    if saved_steps is None:
        candidate_steps = sorted(
            {
                r.step
                for r in held
                if is_saved_checkpoint_step(r.step, save_cadence=save_cadence)
            }
        )
    else:
        candidate_steps = sorted({int(s) for s in saved_steps})

    by_step: Dict[int, StepMetrics] = {}
    for r in held:
        if r.step in by_step and by_step[r.step].as_dict() != r.as_dict():
            raise StatsError(
                f"conflicting metrics for surface={selection_surface!r} "
                f"step={r.step}"
            )
        by_step[r.step] = r

    candidates = [by_step[s] for s in candidate_steps if s in by_step]
    if not candidates:
        raise StatsError(
            "no saved-checkpoint steps have held-out validation metrics; "
            f"saved_steps={candidate_steps}, held_out_steps="
            f"{sorted(by_step)}"
        )

    def _key(r: StepMetrics) -> Tuple[float, int]:
        return (r.metric(primary_metric), r.step)

    best = max(candidates, key=_key)

    if final_step is not None:
        if final_step not in by_step:
            raise StatsError(
                f"final_step={final_step} has no {selection_surface} metrics"
            )
        if not is_saved_checkpoint_step(final_step, save_cadence=save_cadence):
            raise StatsError(
                f"final_step={final_step} is not a saved-checkpoint step "
                f"(cadence {save_cadence})"
            )
        final = by_step[final_step]
    else:
        final = max(candidates, key=lambda r: r.step)

    return BestFinalReport(
        primary_metric=primary_metric,
        selection_surface=selection_surface,
        best=best,
        final=final,
        candidates=tuple(candidates),
    )


def validation_output_dir(
    outputs_root: Path | str,
    run_id: str,
    student_slug: str,
) -> Path:
    return Path(outputs_root) / run_id / student_slug / "validation"


def metrics_jsonl_path(
    outputs_root: Path | str,
    run_id: str,
    student_slug: str,
) -> Path:
    return validation_output_dir(outputs_root, run_id, student_slug) / "metrics.jsonl"


def selection_json_path(
    outputs_root: Path | str,
    run_id: str,
    student_slug: str,
) -> Path:
    return (
        validation_output_dir(outputs_root, run_id, student_slug) / "selection.json"
    )
