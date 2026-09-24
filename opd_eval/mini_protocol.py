"""Mini-protocol eval driver: one HF model dir → (n,c) + metrics jsonl.

Protocol (resume-safe on (problem_id, sample_idx)):

- n=512: H-hard-mini, MATH500-mini, AIME union
- n=16: H-mini non-B0 (B0 reuses H-hard-mini samples), AMC'23, HMMT'25

Does not launch vLLM itself in CPU tests — pass ``generate_fn`` or use
``--dry-plan`` to emit the work list. Live GPU entry:
``python -m opd_eval.mini_protocol --model-dir …``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from adaptors.adaptor_factory import AdaptorFactory
from adaptors.base_adaptor import BaseAdaptor
from opd_eval.sampling import (
    EVAL_SEED_MINI_PROTOCOL,
    EVAL_TEMPERATURE,
    EVAL_TOP_P,
    refuse_binning_seed,
    sample_seed,
)
from opd_eval.stats import (
    mean_unbiased_pass_at_k,
    per_problem_nc,
    pool_from_records,
    unbiased_pass_at_k_curve,
)

GenerateFn = Callable[[str, int], str]  # (prompt, seed) -> response text


@dataclass(frozen=True)
class MiniJob:
    surface: str
    benchmark_id: str
    adaptor_key: str
    data_path: Path
    n: int
    reuse_from: Optional[str] = None  # surface whose B0 samples to reuse
    bin_filter: Optional[str] = None  # if set, only these bins
    bin_exclude: Optional[str] = None


def default_mini_jobs(
    *,
    h_hard_mini: Path,
    h_mini: Path,
    math500_mini: Path,
    aime_union: Path,
    amc23: Path,
    hmmt25: Path,
) -> List[MiniJob]:
    return [
        MiniJob(
            surface="h_hard_mini",
            benchmark_id="h_hard_mini",
            adaptor_key="c1_or1_200",
            data_path=h_hard_mini,
            n=512,
        ),
        MiniJob(
            surface="math500_mini",
            benchmark_id="math500_mini",
            adaptor_key="c1_math500",
            data_path=math500_mini,
            n=512,
        ),
        MiniJob(
            surface="aime_union",
            benchmark_id="aime_union",
            adaptor_key="c1_aime_union",
            data_path=aime_union,
            n=512,
        ),
        MiniJob(
            surface="h_mini",
            benchmark_id="h_mini",
            adaptor_key="c1_or1_200",
            data_path=h_mini,
            n=16,
            reuse_from="h_hard_mini",
            bin_exclude="B0",
        ),
        MiniJob(
            surface="amc23",
            benchmark_id="amc23",
            adaptor_key="c1_amc23",
            data_path=amc23,
            n=16,
        ),
        MiniJob(
            surface="hmmt25",
            benchmark_id="hmmt25",
            adaptor_key="c1_hmmt25",
            data_path=hmmt25,
            n=16,
        ),
    ]


def _load_done_keys(samples_path: Path) -> Set[Tuple[str, int]]:
    done: Set[Tuple[str, int]] = set()
    if not samples_path.is_file():
        return done
    with open(samples_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            done.add((str(rec["problem_id"]), int(rec["sample_idx"])))
    return done


def _append_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def plan_work_items(
    adaptor: BaseAdaptor,
    *,
    n: int,
    base_seed: int,
    done: Set[Tuple[str, int]],
    bin_exclude: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Resume-safe work list: skip completed (problem_id, sample_idx)."""
    refuse_binning_seed(base_seed, context="mini_protocol")
    items: List[Dict[str, Any]] = []
    for item in adaptor.data:
        if bin_exclude is not None and str(item.get("bin", "")) == bin_exclude:
            continue
        pid = adaptor.get_problem_id(item)
        prompt = adaptor.format_prompt(item)
        gt = adaptor.get_ground_truth(item)
        for idx in range(n):
            if (pid, idx) in done:
                continue
            items.append(
                {
                    "problem_id": pid,
                    "sample_idx": idx,
                    "seed": sample_seed(base_seed, pid, idx),
                    "prompt": prompt,
                    "ground_truth": gt,
                    "item": item,
                }
            )
    return items


def score_response(
    adaptor: BaseAdaptor,
    response: str,
    ground_truth: str,
) -> Dict[str, Any]:
    extracted = adaptor.extract_answer(response)
    verified = bool(adaptor.verify_answer(extracted, ground_truth))
    return {
        "response": response,
        "extracted": extracted,
        "verified": verified,
        "n_tokens": len(response.split()),
    }


def write_metrics(
    samples_path: Path,
    metrics_path: Path,
    *,
    n: int,
) -> Dict[str, Any]:
    records = []
    with open(samples_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    pool = pool_from_records(records, n)
    nc = per_problem_nc(pool)
    per_problem = []
    for pid, nc_row in sorted(nc.items()):
        curve = unbiased_pass_at_k_curve(nc_row["n"], nc_row["c"])
        per_problem.append(
            {
                "problem_id": pid,
                "n": nc_row["n"],
                "c": nc_row["c"],
                "pass_at_k": {str(k): v for k, v in curve.items()},
            }
        )
    summary = {
        "n": n,
        "n_problems": len(nc),
        "mean_pass_at_1": mean_unbiased_pass_at_k(pool, n, 1),
        "mean_pass_at_k": {
            str(k): mean_unbiased_pass_at_k(pool, n, k)
            for k in (1, min(8, n), min(16, n), n)
            if k <= n
        },
        "per_problem": per_problem,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    # Also write per-problem nc jsonl for table builders.
    nc_path = metrics_path.with_name("per_problem_nc.jsonl")
    with open(nc_path, "w", encoding="utf-8") as f:
        for row in per_problem:
            f.write(
                json.dumps(
                    {"problem_id": row["problem_id"], "n": row["n"], "c": row["c"]},
                    ensure_ascii=False,
                )
                + "\n"
            )
    return summary


def run_job(
    job: MiniJob,
    *,
    out_root: Path,
    model_dir: Path,
    base_seed: int,
    generate_fn: Optional[GenerateFn] = None,
    dry_plan: bool = False,
) -> Dict[str, Any]:
    refuse_binning_seed(base_seed, context="mini_protocol")
    bench_dir = out_root / job.benchmark_id
    samples_path = bench_dir / "samples.jsonl"
    adaptor = AdaptorFactory.create_adaptor(
        job.adaptor_key, str(job.data_path), thinking_mode=False
    )
    done = _load_done_keys(samples_path)
    work = plan_work_items(
        adaptor,
        n=job.n,
        base_seed=base_seed,
        done=done,
        bin_exclude=job.bin_exclude,
    )
    plan_info = {
        "surface": job.surface,
        "benchmark_id": job.benchmark_id,
        "n": job.n,
        "n_done": len(done),
        "n_todo": len(work),
        "model_dir": str(model_dir),
        "temperature": EVAL_TEMPERATURE,
        "top_p": EVAL_TOP_P,
        "base_seed": base_seed,
    }
    if dry_plan or generate_fn is None:
        return plan_info

    batch: List[Dict[str, Any]] = []
    for w in work:
        text = generate_fn(w["prompt"], int(w["seed"]))
        scored = score_response(adaptor, text, w["ground_truth"])
        batch.append(
            {
                "problem_id": w["problem_id"],
                "sample_idx": w["sample_idx"],
                "seed": w["seed"],
                "benchmark_id": job.benchmark_id,
                "g": 0,
                **scored,
            }
        )
        if len(batch) >= 64:
            _append_jsonl(samples_path, batch)
            batch.clear()
    if batch:
        _append_jsonl(samples_path, batch)

    # B0 reuse note is recorded; actual copy is left to the caller / post-pass
    # when h_hard_mini samples exist (see copy_b0_reuse).
    if job.reuse_from:
        copy_b0_reuse(
            out_root / job.reuse_from / "samples.jsonl",
            samples_path,
            adaptor,
            n=job.n,
        )

    metrics = write_metrics(samples_path, bench_dir / "metrics.json", n=job.n)
    plan_info["metrics"] = {
        "mean_pass_at_1": metrics["mean_pass_at_1"],
        "n_problems": metrics["n_problems"],
    }
    return plan_info


def copy_b0_reuse(
    src_samples: Path,
    dst_samples: Path,
    adaptor: BaseAdaptor,
    *,
    n: int,
) -> int:
    """Copy H-hard-mini sample_idx < n for H-mini B0 problems into dst."""
    b0_ids = {
        adaptor.get_problem_id(item)
        for item in adaptor.data
        if str(item.get("bin", "")) == "B0"
    }
    if not b0_ids or not src_samples.is_file():
        return 0
    done = _load_done_keys(dst_samples)
    copied = 0
    batch: List[Dict[str, Any]] = []
    with open(src_samples, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pid = str(rec["problem_id"])
            idx = int(rec["sample_idx"])
            if pid not in b0_ids or idx >= n:
                continue
            if (pid, idx) in done:
                continue
            batch.append(rec)
            done.add((pid, idx))
            copied += 1
    if batch:
        _append_jsonl(dst_samples, batch)
    return copied


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--base-seed", type=int, default=EVAL_SEED_MINI_PROTOCOL)
    p.add_argument("--h-hard-mini", type=Path, required=True)
    p.add_argument("--h-mini", type=Path, required=True)
    p.add_argument("--math500-mini", type=Path, required=True)
    p.add_argument(
        "--aime-union",
        type=Path,
        default=Path("data/aime24_25_26_bench_schema.jsonl"),
    )
    p.add_argument("--amc23", type=Path, default=Path("data/amc23_bench_schema.jsonl"))
    p.add_argument("--hmmt25", type=Path, default=Path("data/hmmt25_bench_schema.jsonl"))
    p.add_argument(
        "--dry-plan",
        action="store_true",
        help="CPU: emit per-job todo counts; do not call vLLM",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    refuse_binning_seed(int(args.base_seed), context="mini_protocol")
    jobs = default_mini_jobs(
        h_hard_mini=args.h_hard_mini,
        h_mini=args.h_mini,
        math500_mini=args.math500_mini,
        aime_union=args.aime_union,
        amc23=args.amc23,
        hmmt25=args.hmmt25,
    )
    reports = []
    for job in jobs:
        reports.append(
            run_job(
                job,
                out_root=args.out_root,
                model_dir=args.model_dir,
                base_seed=int(args.base_seed),
                generate_fn=None,
                dry_plan=True if args.dry_plan else True,
            )
        )
    # Live vLLM wiring is intentionally not auto-invoked here: the caller
    # supplies generate_fn in-process or uses dry-plan from the CLI until
    # the GPU host wires SamplingParams.seed per request.
    print(json.dumps({"jobs": reports}, indent=2))
    if not args.dry_plan:
        print(
            "NOTE: CLI currently plans only; pass generate_fn in-process "
            "or use --dry-plan. Wire vLLM on the GPU host.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
