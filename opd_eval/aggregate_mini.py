"""Aggregate agent mini-eval compact records into protocol metrics.

Inputs (either):
  - compact per-problem JSONL/CSV from ``docs/eng/eval-data/<model>/``
    (fields: problem_id, bin?, n, c, finish_reason_counts?, mean_n_tokens?)
  - or raw ``samples.jsonl`` (streamed; generations discarded)

Outputs:
  - per-bench summary (pass@k curve points, truncation, mean length)
  - main-table rows (pass@1 from first-16 when n>=16; also n-pool pass@1)
  - figure data (pass@k for k in powers of two up to n)
  - per-bin H breakdown (B0..B3) for h_mini / h_hard_mini

Cross-check: compare overlapping mean_pass_at_k / mean_pass_at_1 against
``metrics.json`` produced by ``mini_protocol.write_metrics``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from opd_eval.stats import StatsError, unbiased_pass_at_k

# Protocol k grid for figure curves: 1,2,4,..., up to n (inclusive of n).
def pass_at_k_grid(n: int) -> List[int]:
    if n < 1:
        return []
    ks: List[int] = []
    k = 1
    while k <= n:
        ks.append(k)
        if k == n:
            break
        nxt = k * 2
        if nxt > n:
            if n not in ks:
                ks.append(n)
            break
        k = nxt
    return ks


# Mini surfaces and their nominal n (protocol).
BENCH_N: Dict[str, int] = {
    "h_hard_mini": 512,
    "math500_mini": 512,
    "aime_union": 512,
    "h_mini": 16,
    "amc23": 16,
    "hmmt25": 16,
}

# H sets that get per-bin breakdown.
H_BENCHES = ("h_mini", "h_hard_mini")

MAIN_TABLE_PASS1_N = 16  # protocol: main-table pass@1 from first 16 samples


@dataclass
class ProblemRecord:
    problem_id: str
    n: int
    c: int
    bin: Optional[str] = None
    mean_n_tokens: Optional[float] = None
    finish_reason_counts: Dict[str, int] = field(default_factory=dict)
    # Main-table pass@1 uses the first 16 samples of the fixed draw.
    n_first16: Optional[int] = None
    c_first16: Optional[int] = None
    # optional: verified bits for first-16 when streaming samples
    verified_prefix: Optional[Tuple[bool, ...]] = None


def _parse_finish_counts(raw: Any) -> Dict[str, int]:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return {str(k): int(v) for k, v in raw.items()}
    if isinstance(raw, str):
        return {str(k): int(v) for k, v in json.loads(raw).items()}
    raise TypeError(f"bad finish_reason_counts: {type(raw)}")


def _optional_int(v: Any) -> Optional[int]:
    if v is None or v == "":
        return None
    return int(v)


def load_compact_jsonl(path: Path) -> List[ProblemRecord]:
    rows: List[ProblemRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append(
                ProblemRecord(
                    problem_id=str(r["problem_id"]),
                    n=int(r["n"]),
                    c=int(r["c"]),
                    bin=(str(r["bin"]) if r.get("bin") not in (None, "") else None),
                    mean_n_tokens=(
                        float(r["mean_n_tokens"])
                        if r.get("mean_n_tokens") is not None
                        else None
                    ),
                    finish_reason_counts=_parse_finish_counts(
                        r.get("finish_reason_counts")
                    ),
                    n_first16=_optional_int(r.get("n_first16")),
                    c_first16=_optional_int(r.get("c_first16")),
                )
            )
    return rows


def load_compact_csv(path: Path) -> List[ProblemRecord]:
    rows: List[ProblemRecord] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(
                ProblemRecord(
                    problem_id=str(r["problem_id"]),
                    n=int(r["n"]),
                    c=int(r["c"]),
                    bin=(str(r["bin"]) if r.get("bin") not in (None, "") else None),
                    mean_n_tokens=(
                        float(r["mean_n_tokens"])
                        if r.get("mean_n_tokens") not in (None, "")
                        else None
                    ),
                    finish_reason_counts=_parse_finish_counts(
                        r.get("finish_reason_counts")
                    ),
                    n_first16=_optional_int(r.get("n_first16")),
                    c_first16=_optional_int(r.get("c_first16")),
                )
            )
    return rows


def stream_samples_to_records(
    samples_path: Path,
    *,
    bin_map: Optional[Mapping[str, str]] = None,
    prefix_n: int = MAIN_TABLE_PASS1_N,
) -> Tuple[List[ProblemRecord], Dict[str, Any]]:
    """Stream samples.jsonl → ProblemRecord list; defensive (pid, idx) dedup."""
    n_correct: Dict[str, int] = defaultdict(int)
    n_samples: Dict[str, int] = defaultdict(int)
    tok_sum: Dict[str, float] = defaultdict(float)
    fr_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    prefix_bits: Dict[str, Dict[int, bool]] = defaultdict(dict)
    seen: Set[Tuple[str, int]] = set()
    dups: List[Tuple[str, int]] = []
    n_lines = 0

    with open(samples_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            r = json.loads(line)
            pid = str(r["problem_id"])
            idx = int(r["sample_idx"])
            key = (pid, idx)
            if key in seen:
                dups.append(key)
                continue
            seen.add(key)
            n_samples[pid] += 1
            verified = bool(r.get("verified"))
            if verified:
                n_correct[pid] += 1
            tok_sum[pid] += float(r.get("n_tokens") or 0)
            fr = str(r.get("finish_reason") or "unknown")
            fr_counts[pid][fr] += 1
            if idx < prefix_n:
                prefix_bits[pid][idx] = verified

    rows: List[ProblemRecord] = []
    for pid in sorted(n_samples.keys()):
        n = n_samples[pid]
        bits = prefix_bits.get(pid, {})
        verified_prefix = None
        n16 = c16 = None
        need = min(prefix_n, n)
        if bits and all(i in bits for i in range(need)):
            verified_prefix = tuple(bits[i] for i in range(need))
            n16 = need
            c16 = sum(1 for v in verified_prefix if v)
        rows.append(
            ProblemRecord(
                problem_id=pid,
                n=n,
                c=n_correct[pid],
                bin=(bin_map or {}).get(pid),
                mean_n_tokens=(tok_sum[pid] / n) if n else 0.0,
                finish_reason_counts=dict(fr_counts[pid]),
                n_first16=n16,
                c_first16=c16,
                verified_prefix=verified_prefix,
            )
        )
    meta = {
        "n_lines": n_lines,
        "n_kept": n_lines - len(dups),
        "n_dup_keys": len(dups),
        "dup_keys_sample": [list(k) for k in dups[:20]],
    }
    return rows, meta


def merge_b0_from_hard(
    h_mini: Sequence[ProblemRecord],
    h_hard: Sequence[ProblemRecord],
    *,
    b0_ids: Set[str],
    n_reuse: int = MAIN_TABLE_PASS1_N,
) -> Tuple[List[ProblemRecord], Dict[str, Any]]:
    """Protocol: H-mini B0 reuses H-hard-mini samples (first ``n_reuse``).

    Keeps existing h_mini rows (including already-copied B0). Fills only
    missing B0 ids from hard via ``n_first16``/``c_first16``, verified_prefix,
    or hard rows that already have ``n == n_reuse``.
    """
    by_pid = {r.problem_id: r for r in h_mini}
    hard_by = {r.problem_id: r for r in h_hard}
    missing = [pid for pid in sorted(b0_ids) if pid not in by_pid]
    reused = 0
    for pid in missing:
        hard = hard_by.get(pid)
        if hard is None:
            raise StatsError(f"B0 problem {pid!r} missing from both h_mini and h_hard")
        if hard.n_first16 is not None and hard.c_first16 is not None:
            n16, c16 = int(hard.n_first16), int(hard.c_first16)
            if n16 != n_reuse:
                raise StatsError(
                    f"B0 reuse {pid!r}: hard n_first16={n16} != {n_reuse}"
                )
            rec = ProblemRecord(
                problem_id=pid,
                n=n16,
                c=c16,
                bin="B0",
                mean_n_tokens=hard.mean_n_tokens,
                finish_reason_counts=hard.finish_reason_counts,
                n_first16=n16,
                c_first16=c16,
                verified_prefix=hard.verified_prefix[:n_reuse]
                if hard.verified_prefix is not None
                else None,
            )
        elif hard.verified_prefix is not None:
            pref = hard.verified_prefix[:n_reuse]
            rec = ProblemRecord(
                problem_id=pid,
                n=len(pref),
                c=sum(1 for v in pref if v),
                bin="B0",
                mean_n_tokens=hard.mean_n_tokens,
                finish_reason_counts=hard.finish_reason_counts,
                n_first16=len(pref),
                c_first16=sum(1 for v in pref if v),
                verified_prefix=pref,
            )
        elif hard.n == n_reuse:
            rec = ProblemRecord(
                problem_id=pid,
                n=hard.n,
                c=hard.c,
                bin="B0",
                mean_n_tokens=hard.mean_n_tokens,
                finish_reason_counts=hard.finish_reason_counts,
                n_first16=hard.n,
                c_first16=hard.c,
            )
        else:
            raise StatsError(
                f"cannot reuse B0 {pid!r} from hard n={hard.n} without "
                f"n_first16/c_first16 or verified_prefix"
            )
        by_pid[pid] = rec
        reused += 1
    return [by_pid[k] for k in sorted(by_pid.keys())], {
        "b0_ids": len(b0_ids),
        "reused": reused,
        "already_present": len(b0_ids) - reused,
    }


def mean_unbiased_pass_at_k_from_nc(
    records: Sequence[ProblemRecord], k: int
) -> float:
    if not records:
        raise StatsError("empty records")
    total = 0.0
    for r in records:
        total += unbiased_pass_at_k(r.n, r.c, k)
    return total / len(records)


def pass_at_1_first16(records: Sequence[ProblemRecord]) -> float:
    """Main-table pass@1: unbiased from first 16 samples (or all if n<=16)."""
    if not records:
        raise StatsError("empty records")
    vals: List[float] = []
    for r in records:
        if r.n_first16 is not None and r.c_first16 is not None:
            vals.append(unbiased_pass_at_k(int(r.n_first16), int(r.c_first16), 1))
        elif r.verified_prefix is not None:
            pref = r.verified_prefix[:MAIN_TABLE_PASS1_N]
            vals.append(unbiased_pass_at_k(len(pref), sum(1 for v in pref if v), 1))
        elif r.n <= MAIN_TABLE_PASS1_N:
            vals.append(unbiased_pass_at_k(r.n, r.c, 1))
        else:
            raise StatsError(
                f"{r.problem_id}: n={r.n}>16 without n_first16/c_first16; "
                "cannot compute main-table first-16 pass@1"
            )
    return sum(vals) / len(vals)


def truncation_rate(records: Sequence[ProblemRecord]) -> Optional[float]:
    total = 0
    trunc = 0
    any_fr = False
    for r in records:
        fr = r.finish_reason_counts or {}
        if not fr:
            continue
        any_fr = True
        for reason, cnt in fr.items():
            total += int(cnt)
            if reason == "length":
                trunc += int(cnt)
    if not any_fr or total == 0:
        return None
    return trunc / total


def mean_length(records: Sequence[ProblemRecord]) -> Optional[float]:
    vals = [r.mean_n_tokens for r in records if r.mean_n_tokens is not None]
    if not vals:
        return None
    # Weight by n so problems with equal n are equally weighted; here each
    # record already stores mean over its samples → mean of means = overall
    # if n equal; weight by n for safety.
    num = 0.0
    den = 0
    for r in records:
        if r.mean_n_tokens is None:
            continue
        num += r.mean_n_tokens * r.n
        den += r.n
    return num / den if den else None


@dataclass
class BenchAggregate:
    benchmark: str
    n_problems: int
    n_per_problem: int
    pass_at_k: Dict[int, float]
    pass_at_1_main: Optional[float]  # first-16
    pass_at_1_pool: float  # from full n
    truncation_rate: Optional[float]
    mean_n_tokens: Optional[float]
    by_bin: Dict[str, Dict[str, Any]] = field(default_factory=dict)


def aggregate_bench(
    benchmark: str,
    records: Sequence[ProblemRecord],
    *,
    expect_n: Optional[int] = None,
) -> BenchAggregate:
    if not records:
        raise StatsError(f"{benchmark}: empty")
    ns = {r.n for r in records}
    if len(ns) != 1:
        raise StatsError(f"{benchmark}: inconsistent n across problems: {ns}")
    n = next(iter(ns))
    if expect_n is not None and n != expect_n:
        raise StatsError(f"{benchmark}: expected n={expect_n}, got {n}")

    ks = pass_at_k_grid(n)
    curve = {k: mean_unbiased_pass_at_k_from_nc(records, k) for k in ks}
    pool_p1 = mean_unbiased_pass_at_k_from_nc(records, 1)

    main_p1: Optional[float]
    try:
        main_p1 = pass_at_1_first16(records)
    except StatsError:
        # For n>16 compact-only inputs: main-table first-16 unavailable.
        # Exchangeability: pool pass@1 == c/n equals unbiased first-16 in
        # expectation; protocol still wants first-16 of the fixed draw.
        # Leave None so callers know; CLI can fall back when streaming.
        main_p1 = None if n > MAIN_TABLE_PASS1_N else pool_p1

    by_bin: Dict[str, Dict[str, Any]] = {}
    bins_present = {r.bin for r in records if r.bin}
    if bins_present:
        for b in sorted(bins_present):
            subset = [r for r in records if r.bin == b]
            by_bin[b] = {
                "n_problems": len(subset),
                "n_per_problem": n,
                "pass_at_k": {
                    k: mean_unbiased_pass_at_k_from_nc(subset, k) for k in ks
                },
                "pass_at_1_pool": mean_unbiased_pass_at_k_from_nc(subset, 1),
                "truncation_rate": truncation_rate(subset),
                "mean_n_tokens": mean_length(subset),
            }
            try:
                by_bin[b]["pass_at_1_main"] = pass_at_1_first16(subset)
            except StatsError:
                by_bin[b]["pass_at_1_main"] = None

    return BenchAggregate(
        benchmark=benchmark,
        n_problems=len(records),
        n_per_problem=n,
        pass_at_k=curve,
        pass_at_1_main=main_p1,
        pass_at_1_pool=pool_p1,
        truncation_rate=truncation_rate(records),
        mean_n_tokens=mean_length(records),
        by_bin=by_bin,
    )


def cross_check_metrics(
    agg: BenchAggregate,
    metrics: Mapping[str, Any],
    *,
    atol: float = 1e-12,
) -> List[Dict[str, Any]]:
    """Compare aggregator vs mini_protocol metrics.json overlapping fields."""
    diffs: List[Dict[str, Any]] = []

    def _add(field: str, a: float, b: float) -> None:
        if not math.isclose(a, b, rel_tol=0.0, abs_tol=atol):
            diffs.append(
                {
                    "field": field,
                    "aggregate": a,
                    "metrics_json": b,
                    "delta": a - b,
                }
            )

    if "mean_pass_at_1" in metrics:
        _add("mean_pass_at_1", agg.pass_at_1_pool, float(metrics["mean_pass_at_1"]))

    mpk = metrics.get("mean_pass_at_k") or {}
    for k_str, v in mpk.items():
        k = int(k_str)
        if k in agg.pass_at_k:
            _add(f"mean_pass_at_k[{k}]", agg.pass_at_k[k], float(v))
        else:
            # metrics may have k not on power-of-two grid; compute on the fly
            # from stored n if we can — skip if not in curve
            diffs.append(
                {
                    "field": f"mean_pass_at_k[{k}]",
                    "aggregate": None,
                    "metrics_json": float(v),
                    "delta": None,
                    "note": "k not on figure grid; skipped exact compare",
                }
            )
    return diffs


def cross_check_metrics_full(
    records: Sequence[ProblemRecord],
    metrics: Mapping[str, Any],
    *,
    atol: float = 1e-12,
) -> List[Dict[str, Any]]:
    """Compare all keys in metrics.mean_pass_at_k against fresh (n,c) means."""
    diffs: List[Dict[str, Any]] = []
    if "mean_pass_at_1" in metrics:
        a = mean_unbiased_pass_at_k_from_nc(records, 1)
        b = float(metrics["mean_pass_at_1"])
        if not math.isclose(a, b, rel_tol=0.0, abs_tol=atol):
            diffs.append(
                {"field": "mean_pass_at_1", "aggregate": a, "metrics_json": b, "delta": a - b}
            )
    for k_str, v in (metrics.get("mean_pass_at_k") or {}).items():
        k = int(k_str)
        a = mean_unbiased_pass_at_k_from_nc(records, k)
        b = float(v)
        if not math.isclose(a, b, rel_tol=0.0, abs_tol=atol):
            diffs.append(
                {
                    "field": f"mean_pass_at_k[{k}]",
                    "aggregate": a,
                    "metrics_json": b,
                    "delta": a - b,
                }
            )
    n_met = metrics.get("n_problems")
    if n_met is not None and int(n_met) != len(records):
        diffs.append(
            {
                "field": "n_problems",
                "aggregate": len(records),
                "metrics_json": int(n_met),
                "delta": len(records) - int(n_met),
            }
        )
    return diffs


def tidy_main_table_rows(
    *,
    size: str,
    method: str,
    checkpoint: str,
    agg: BenchAggregate,
    notes: str = "",
) -> List[Dict[str, Any]]:
    """Long-format main-table rows matching agent_mini sheet headers."""
    rows = []
    # Primary main-table metric: pass@1 from first 16 when available
    p1 = agg.pass_at_1_main if agg.pass_at_1_main is not None else agg.pass_at_1_pool
    n_for_main = MAIN_TABLE_PASS1_N if agg.n_per_problem >= MAIN_TABLE_PASS1_N else agg.n_per_problem
    rows.append(
        {
            "size": size,
            "method": method,
            "checkpoint": checkpoint,
            "benchmark": agg.benchmark,
            "n_problems": agg.n_problems,
            "n_samples_per_problem": n_for_main,
            "metric": "pass@1",
            "value": p1,
            "notes": notes
            or (
                "first-16 unbiased"
                if agg.pass_at_1_main is not None
                else "pool unbiased (n<=16 or fallback)"
            ),
        }
    )
    if agg.n_per_problem > MAIN_TABLE_PASS1_N:
        rows.append(
            {
                "size": size,
                "method": method,
                "checkpoint": checkpoint,
                "benchmark": agg.benchmark,
                "n_problems": agg.n_problems,
                "n_samples_per_problem": agg.n_per_problem,
                "metric": "pass@1_pool",
                "value": agg.pass_at_1_pool,
                "notes": f"unbiased from full n={agg.n_per_problem}",
            }
        )
        rows.append(
            {
                "size": size,
                "method": method,
                "checkpoint": checkpoint,
                "benchmark": agg.benchmark,
                "n_problems": agg.n_problems,
                "n_samples_per_problem": agg.n_per_problem,
                "metric": f"pass@{agg.n_per_problem}",
                "value": agg.pass_at_k[agg.n_per_problem],
                "notes": "frontier pass@max",
            }
        )
    else:
        kmax = agg.n_per_problem
        rows.append(
            {
                "size": size,
                "method": method,
                "checkpoint": checkpoint,
                "benchmark": agg.benchmark,
                "n_problems": agg.n_problems,
                "n_samples_per_problem": kmax,
                "metric": f"pass@{kmax}",
                "value": agg.pass_at_k[kmax],
                "notes": "pass@max",
            }
        )
    if agg.truncation_rate is not None:
        rows.append(
            {
                "size": size,
                "method": method,
                "checkpoint": checkpoint,
                "benchmark": agg.benchmark,
                "n_problems": agg.n_problems,
                "n_samples_per_problem": agg.n_per_problem,
                "metric": "truncation_rate",
                "value": agg.truncation_rate,
                "notes": "finish_reason==length",
            }
        )
    if agg.mean_n_tokens is not None:
        rows.append(
            {
                "size": size,
                "method": method,
                "checkpoint": checkpoint,
                "benchmark": agg.benchmark,
                "n_problems": agg.n_problems,
                "n_samples_per_problem": agg.n_per_problem,
                "metric": "mean_n_tokens",
                "value": agg.mean_n_tokens,
                "notes": "",
            }
        )
    return rows


def tidy_figure_rows(
    *,
    size: str,
    method: str,
    checkpoint: str,
    agg: BenchAggregate,
    notes: str = "mini single draw; no seed repeats",
) -> List[Dict[str, Any]]:
    rows = []
    for k, v in sorted(agg.pass_at_k.items()):
        rows.append(
            {
                "size": size,
                "method": method,
                "checkpoint": checkpoint,
                "benchmark": agg.benchmark,
                "k": k,
                "n_samples_per_problem": agg.n_per_problem,
                "repeat_id": 0,
                "pass_at_k": v,
                "notes": notes,
            }
        )
    return rows


def tidy_per_bin_rows(
    *,
    size: str,
    method: str,
    checkpoint: str,
    agg: BenchAggregate,
) -> List[Dict[str, Any]]:
    rows = []
    for b, info in sorted(agg.by_bin.items()):
        p1 = info.get("pass_at_1_main")
        if p1 is None:
            p1 = info["pass_at_1_pool"]
        rows.append(
            {
                "size": size,
                "method": method,
                "checkpoint": checkpoint,
                "benchmark": agg.benchmark,
                "bin": b,
                "n_problems": info["n_problems"],
                "n_samples_per_problem": info["n_per_problem"],
                "pass_at_1": p1,
                "pass_at_max": info["pass_at_k"][info["n_per_problem"]],
                "truncation_rate": info.get("truncation_rate"),
                "mean_n_tokens": info.get("mean_n_tokens"),
            }
        )
        for k, v in sorted(info["pass_at_k"].items()):
            rows.append(
                {
                    "size": size,
                    "method": method,
                    "checkpoint": checkpoint,
                    "benchmark": agg.benchmark,
                    "bin": b,
                    "n_problems": info["n_problems"],
                    "n_samples_per_problem": info["n_per_problem"],
                    "k": k,
                    "pass_at_k": v,
                }
            )
    return rows


def tidy_bench_summary(agg: BenchAggregate) -> Dict[str, Any]:
    return {
        "benchmark": agg.benchmark,
        "n_problems": agg.n_problems,
        "n_per_problem": agg.n_per_problem,
        "pass_at_1_main": agg.pass_at_1_main,
        "pass_at_1_pool": agg.pass_at_1_pool,
        "pass_at_max": agg.pass_at_k[agg.n_per_problem],
        "truncation_rate": agg.truncation_rate,
        "mean_n_tokens": agg.mean_n_tokens,
        "pass_at_k": {str(k): v for k, v in sorted(agg.pass_at_k.items())},
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    # union of keys, stable order from first row then extras
    keys: List[str] = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(dict(r))


def load_records_for_bench(data_root: Path, bench: str) -> List[ProblemRecord]:
    jsonl = data_root / bench / f"{bench}_per_problem.jsonl"
    if not jsonl.is_file():
        jsonl = data_root / f"{bench}_per_problem.jsonl"
    if jsonl.is_file():
        return load_compact_jsonl(jsonl)
    csv_path = data_root / bench / f"{bench}_per_problem.csv"
    if not csv_path.is_file():
        csv_path = data_root / f"{bench}_per_problem.csv"
    if csv_path.is_file():
        return load_compact_csv(csv_path)
    raise FileNotFoundError(f"no compact records for {bench} under {data_root}")


def load_metrics_for_bench(data_root: Path, bench: str) -> Optional[Dict[str, Any]]:
    for p in (
        data_root / bench / "metrics.json",
        data_root / f"{bench}_metrics.json",
    ):
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


def run_model_dir(
    data_root: Path,
    *,
    size: str = "1.7B",
    method: str = "Base",
    checkpoint: str = "base",
    out_dir: Optional[Path] = None,
    benches: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Aggregate all benches under a compact data root; write tidy CSVs."""
    out_dir = out_dir or (data_root / "aggregate")
    out_dir.mkdir(parents=True, exist_ok=True)
    benches = list(benches or BENCH_N.keys())

    summaries = []
    main_rows: List[Dict[str, Any]] = []
    fig_rows: List[Dict[str, Any]] = []
    bin_rows: List[Dict[str, Any]] = []
    cross: Dict[str, Any] = {}

    aggs: Dict[str, BenchAggregate] = {}
    for bench in benches:
        records = load_records_for_bench(data_root, bench)
        expect = BENCH_N.get(bench)
        agg = aggregate_bench(bench, records, expect_n=expect)
        aggs[bench] = agg
        summaries.append(tidy_bench_summary(agg))
        main_rows.extend(
            tidy_main_table_rows(
                size=size, method=method, checkpoint=checkpoint, agg=agg
            )
        )
        fig_rows.extend(
            tidy_figure_rows(
                size=size, method=method, checkpoint=checkpoint, agg=agg
            )
        )
        if bench in H_BENCHES and agg.by_bin:
            bin_rows.extend(
                tidy_per_bin_rows(
                    size=size, method=method, checkpoint=checkpoint, agg=agg
                )
            )

        metrics = load_metrics_for_bench(data_root, bench)
        if metrics is not None:
            diffs = cross_check_metrics_full(records, metrics)
            cross[bench] = {
                "n_diffs": len(diffs),
                "diffs": diffs,
                "ok": len(diffs) == 0,
            }

    _write_csv(out_dir / "bench_summary.csv", summaries)
    _write_csv(out_dir / "main_table.csv", main_rows)
    _write_csv(out_dir / "figure_pass_at_k.csv", fig_rows)
    _write_csv(out_dir / "per_bin_h.csv", bin_rows)
    (out_dir / "cross_check.json").write_text(
        json.dumps(cross, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "summaries.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "out_dir": str(out_dir),
        "summaries": summaries,
        "cross_check": cross,
        "n_main_rows": len(main_rows),
        "n_fig_rows": len(fig_rows),
        "n_bin_rows": len(bin_rows),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Compact eval-data dir (e.g. docs/eng/eval-data/base_1p7b)",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--size", default="1.7B")
    p.add_argument("--method", default="Base")
    p.add_argument("--checkpoint", default="base")
    p.add_argument(
        "--benches",
        nargs="*",
        default=None,
        help="Subset of benches (default: all six)",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_model_dir(
        args.data_root,
        size=args.size,
        method=args.method,
        checkpoint=args.checkpoint,
        out_dir=args.out_dir,
        benches=args.benches,
    )
    print(json.dumps({k: result[k] for k in ("out_dir", "cross_check", "n_main_rows", "n_fig_rows", "n_bin_rows")}, indent=2))
    for s in result["summaries"]:
        print(
            f"{s['benchmark']}: pass@1_pool={s['pass_at_1_pool']:.6f} "
            f"pass@max={s['pass_at_max']:.6f} trunc={s['truncation_rate']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
