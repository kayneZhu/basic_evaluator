"""Pre-registered large-k boundary report.

Unbiased pass@k (Chen et al. 2021) for
``k in {1,2,4,8,16,32,64,128,256,512}`` with ``k <= n``, pooled and by
subset. AIME year bins are also pooled as ``AIME``. The ``amc23`` bin
is reported as ``AMC23``. On the n=256 H-mini tables the subsets are
``B0`` and ``B1``.

Difference versus the base file is a paired bootstrap over problems,
stratified by ``bin``, 2000 resamples by default. SE is the sample
standard deviation (ddof=1) of the resampled differences. The same
problem draws are reused for every k.

Solvable means at least one correct sample (the pass@n bit). Learned:
base unsolved and model solved. Forgotten: the opposite. Crossing k is
the smallest reported k whose (model − base) mean is negative.

Usage::

    python -m opd_eval.boundary_report \\
        --base docs/eng/reports/data/c2_base/boundary_g1_base.jsonl.gz \\
        --model g2=docs/eng/reports/data/c2_base/boundary_g2_s300.jsonl.gz \\
        --model g3=docs/eng/reports/data/c2_base/boundary_g3_s300.jsonl.gz \\
        --boot 2000 --seed 20261005
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from opd_eval.stats import unbiased_pass_at_k

REGISTERED_KS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
DEFAULT_BOOT = 2000
DEFAULT_SEED = 20261005
# Ignore float dust when deciding that a difference is negative.
NEG_EPS = 1e-9


class ReportError(ValueError):
    pass


def load_problems(path: Path) -> List[Dict[str, Any]]:
    """One record per problem: bin, n, c, trunc count, token sum."""
    acc: Dict[str, Dict[str, Any]] = {}
    seen = set()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pid = str(row["problem_id"])
            idx = int(row["sample_idx"])
            key = (pid, idx)
            if key in seen:
                raise ReportError(f"duplicate sample {key} in {path}")
            seen.add(key)
            rec = acc.get(pid)
            if rec is None:
                rec = {
                    "problem_id": pid,
                    "bin": str(row["bin"]),
                    "n": 0,
                    "c": 0,
                    "n_trunc": 0,
                    "sum_tokens": 0,
                    "idxs": [],
                }
                acc[pid] = rec
            elif rec["bin"] != str(row["bin"]):
                raise ReportError(f"bin mismatch for {pid} in {path}")
            rec["n"] += 1
            rec["c"] += int(bool(row["correct"]))
            rec["n_trunc"] += int(bool(row["truncated"]))
            rec["sum_tokens"] += int(row["n_tokens"])
            rec["idxs"].append(idx)
    problems = []
    for pid in sorted(acc):
        rec = acc[pid]
        idxs = sorted(rec["idxs"])
        n = int(rec["n"])
        if idxs != list(range(n)):
            raise ReportError(f"{pid} sample_idx is not 0..{n - 1}")
        problems.append(
            {
                "problem_id": pid,
                "bin": rec["bin"],
                "n": n,
                "c": int(rec["c"]),
                "n_trunc": int(rec["n_trunc"]),
                "sum_tokens": int(rec["sum_tokens"]),
            }
        )
    if not problems:
        raise ReportError(f"no rows in {path}")
    n0 = problems[0]["n"]
    if any(p["n"] != n0 for p in problems):
        raise ReportError(f"uneven n in {path}")
    return problems


def _align(base: Sequence[Mapping[str, Any]], other: Sequence[Mapping[str, Any]], name: str):
    by_id = {row["problem_id"]: row for row in other}
    if set(by_id) != {row["problem_id"] for row in base}:
        raise ReportError(f"{name} problem ids differ from base")
    aligned = []
    for row in base:
        got = by_id[row["problem_id"]]
        if got["bin"] != row["bin"] or int(got["n"]) != int(row["n"]):
            raise ReportError(f"{name} bin/n mismatch on {row['problem_id']}")
        aligned.append(got)
    return aligned


def scope_indices(problems: Sequence[Mapping[str, Any]]) -> Dict[str, List[int]]:
    """Pooled, each bin, and ``AIME`` when year bins are present."""
    scopes: Dict[str, List[int]] = {
        "pooled": list(range(len(problems))),
    }
    by_bin: Dict[str, List[int]] = defaultdict(list)
    for i, row in enumerate(problems):
        by_bin[str(row["bin"])].append(i)
    for label in sorted(by_bin):
        # Pre-registered subset name for the AMC file. Year bins stay aime24/25/26.
        key = "AMC23" if label == "amc23" else label
        scopes[key] = by_bin[label]
    aime = [i for i, row in enumerate(problems) if str(row["bin"]).startswith("aime")]
    if aime:
        scopes["AIME"] = aime
    return scopes


def _curves(problems: Sequence[Mapping[str, Any]], indices: Sequence[int], ks: Sequence[int]):
    return [
        [unbiased_pass_at_k(int(problems[i]["n"]), int(problems[i]["c"]), int(k)) for k in ks]
        for i in indices
    ]


def _mean_curve(curves: Sequence[Sequence[float]]) -> List[float]:
    n = len(curves)
    return [sum(row[j] for row in curves) / n for j in range(len(curves[0]))]


def _stratified_draws(bins: Sequence[str], n_boot: int, rng: random.Random) -> List[List[int]]:
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, label in enumerate(bins):
        groups[label].append(i)
    draws = []
    for _ in range(int(n_boot)):
        draw: List[int] = []
        for label in sorted(groups):
            pool = groups[label]
            m = len(pool)
            draw.extend(pool[rng.randrange(m)] for _ in range(m))
        draws.append(draw)
    return draws


def _scope_payload(
    problems: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    ks: Sequence[int],
) -> Dict[str, Any]:
    curves = _curves(problems, indices, ks)
    means = _mean_curve(curves)
    solvable = sum(1 for i in indices if int(problems[i]["c"]) > 0)
    samples = sum(int(problems[i]["n"]) for i in indices)
    trunc = sum(int(problems[i]["n_trunc"]) for i in indices)
    tokens = sum(int(problems[i]["sum_tokens"]) for i in indices)
    return {
        "n_problems": len(indices),
        "n": int(problems[indices[0]]["n"]),
        "pass_at_k": {str(k): means[j] for j, k in enumerate(ks)},
        "solvable": int(solvable),
        "trunc_rate": trunc / samples,
        "mean_len": tokens / samples,
        "_curves": curves,
        "_bins": [str(problems[i]["bin"]) for i in indices],
        "_c": [int(problems[i]["c"]) for i in indices],
    }


def _diff_block(
    base_scope: Mapping[str, Any],
    model_scope: Mapping[str, Any],
    ks: Sequence[int],
    draws: Sequence[Sequence[int]],
) -> Dict[str, Any]:
    base_curves = base_scope["_curves"]
    model_curves = model_scope["_curves"]
    diff: Dict[str, float] = {}
    se: Dict[str, float] = {}
    for j, k in enumerate(ks):
        point = float(model_scope["pass_at_k"][str(k)]) - float(base_scope["pass_at_k"][str(k)])
        deltas = []
        width = len(draws[0])
        for draw in draws:
            base_mean = sum(base_curves[i][j] for i in draw) / width
            model_mean = sum(model_curves[i][j] for i in draw) / width
            deltas.append(model_mean - base_mean)
        diff[str(k)] = point
        se[str(k)] = float(statistics.stdev(deltas))
    crossing = None
    for k in ks:
        if diff[str(k)] < -NEG_EPS:
            crossing = int(k)
            break
    base_c = base_scope["_c"]
    model_c = model_scope["_c"]
    learned = forgotten = base_solved = base_unsolved = 0
    for b_c, m_c in zip(base_c, model_c):
        b_ok = b_c > 0
        m_ok = m_c > 0
        if b_ok:
            base_solved += 1
        else:
            base_unsolved += 1
        if (not b_ok) and m_ok:
            learned += 1
        elif b_ok and (not m_ok):
            forgotten += 1
    return {
        "diff": diff,
        "se": se,
        "crossing_k": crossing,
        "learned": learned,
        "forgotten": forgotten,
        "base_solved": base_solved,
        "base_unsolved": base_unsolved,
    }


def _public_scope(scope: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in scope.items() if not key.startswith("_")}


def build_report(
    base_path: Path,
    models: Mapping[str, Path],
    *,
    n_boot: int = DEFAULT_BOOT,
    seed: int = DEFAULT_SEED,
    ks: Sequence[int] = REGISTERED_KS,
) -> Dict[str, Any]:
    if int(n_boot) < 2:
        raise ReportError("bootstrap needs at least 2 resamples")
    base = load_problems(Path(base_path))
    n = int(base[0]["n"])
    use_ks = [int(k) for k in ks if int(k) <= n]
    if not use_ks:
        raise ReportError(f"no k <= n={n}")
    aligned = {name: _align(base, load_problems(Path(path)), name) for name, path in models.items()}
    scopes = scope_indices(base)
    rng = random.Random(int(seed))
    # One draw list per scope, shared by every model and every k.
    draws = {
        name: _stratified_draws(
            [str(base[i]["bin"]) for i in indices],
            int(n_boot),
            rng,
        )
        for name, indices in scopes.items()
    }
    base_scopes = {name: _scope_payload(base, indices, use_ks) for name, indices in scopes.items()}
    model_scopes: Dict[str, Dict[str, Any]] = {}
    versus: Dict[str, Dict[str, Any]] = {}
    for name, rows in aligned.items():
        packed = {scope: _scope_payload(rows, indices, use_ks) for scope, indices in scopes.items()}
        versus[name] = {
            scope: _diff_block(base_scopes[scope], packed[scope], use_ks, draws[scope])
            for scope in scopes
        }
        model_scopes[name] = {scope: _public_scope(packed[scope]) for scope in scopes}
    return {
        "n": n,
        "ks": use_ks,
        "ks_omitted": [int(k) for k in ks if int(k) > n],
        "boot": int(n_boot),
        "seed": int(seed),
        "stratified_by": "bin",
        "base": {scope: _public_scope(base_scopes[scope]) for scope in scopes},
        "models": model_scopes,
        "vs_base": versus,
    }


def format_report(report: Mapping[str, Any]) -> str:
    ks = [int(k) for k in report["ks"]]
    lines = [
        f"n={report['n']} boot={report['boot']} seed={report['seed']} "
        f"ks={ks} omitted={report['ks_omitted']}",
    ]
    names = ["base", *report["models"].keys()]
    blocks = {"base": report["base"], **report["models"]}
    preferred = ["pooled", "AIME", "AMC23", "B0", "B1", "B2", "B3"]
    scope_names = list(report["base"])
    ordered = [name for name in preferred if name in scope_names]
    ordered.extend(name for name in scope_names if name not in ordered)
    for scope in ordered:
        lines.append(f"## {scope}  problems={report['base'][scope]['n_problems']}")
        header = "model " + " ".join(f"@{k}" for k in ks) + " solved trunc mean_len"
        lines.append(header)
        for name in names:
            block = blocks[name][scope]
            cells = " ".join(f"{block['pass_at_k'][str(k)]:.4f}" for k in ks)
            lines.append(
                f"{name} {cells} {block['solvable']}/{block['n_problems']} "
                f"{block['trunc_rate']:.4f} {block['mean_len']:.1f}"
            )
        for name, versus in report["vs_base"].items():
            diff = versus[scope]
            cells = " ".join(
                f"{diff['diff'][str(k)]:+.4f}({diff['se'][str(k)]:.4f})" for k in ks
            )
            lines.append(
                f"{name}-base {cells} cross={diff['crossing_k']} "
                f"learned={diff['learned']}/{diff['base_unsolved']} "
                f"forgotten={diff['forgotten']}/{diff['base_solved']}"
            )
    return "\n".join(lines) + "\n"


def _parse_model(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ReportError(f"--model must be name=path, got {spec}")
    name, raw = spec.split("=", 1)
    name = name.strip()
    if not name:
        raise ReportError(f"empty model name in {spec}")
    return name, Path(raw)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True, help="name=jsonl.gz")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--boot", type=int, default=DEFAULT_BOOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--ks", default=",".join(str(k) for k in REGISTERED_KS))
    args = parser.parse_args(argv)
    models = {}
    for spec in args.model:
        name, path = _parse_model(spec)
        if name in models:
            raise SystemExit(f"duplicate model name {name}")
        models[name] = path
    ks = [int(part) for part in str(args.ks).split(",") if part.strip()]
    report = build_report(
        args.base,
        models,
        n_boot=int(args.boot),
        seed=int(args.seed),
        ks=ks,
    )
    text = format_report(report)
    print(text, end="")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
