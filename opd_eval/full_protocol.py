"""Full-set eval plan. Replaces the mini surfaces in the eval queue.

Main table is pass@1. H-hard, MATH500, and the AIME union are drawn once
at n=512 (Fig 4, one seed). Their pass@1 is that same sample, not a second
n=16 run. H, AMC'23, and HMMT'25 Feb stay at n=16. H at K=128 is added
only for the Phase-1 checkpoints.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional, Sequence

from opd_eval.mini_protocol import (
    MiniJob,
    _plan_run_total,
    count_run_samples,
    run_job,
    write_progress,
)
from opd_eval.sampling import EVAL_SEED_MINI_PROTOCOL, refuse_binning_seed

# First Fig 4 seed. Binning seed 20260924 and eval seeds {0, 1} stay unused.
FULL_PROTOCOL_SEED = EVAL_SEED_MINI_PROTOCOL
H_K128_RUNS = frozenset({"C00", "C00p", "C00'", "v5_C01", "C01", "v5_C01g", "C01g", "v5_C01f", "C01f"})
FIG4_N = 512
PASS1_N = 16
H_K = 128
H_MINI_N = 512

DEFAULT_H = Path("/root/autodl-tmp/data/processed/v5_20260927/h400.jsonl")
DEFAULT_MATH500 = Path("data/math500_bench_schema.jsonl")
DEFAULT_AIME = Path("data/aime24_25_26_bench_schema.jsonl")
DEFAULT_AMC = Path("data/amc23_bench_schema.jsonl")
DEFAULT_HMMT = Path("data/hmmt25_bench_schema.jsonl")


def build_full_jobs(
    *,
    h_path: Path = DEFAULT_H,
    math500: Path = DEFAULT_MATH500,
    aime_union: Path = DEFAULT_AIME,
    amc23: Path = DEFAULT_AMC,
    hmmt25: Path = DEFAULT_HMMT,
    include_h_k128: bool = False,
) -> list[MiniJob]:
    """Generation jobs. Fig 4 surfaces are n=512 only."""
    jobs = [
        MiniJob(
            surface="h",
            benchmark_id="h",
            adaptor_key="c1_or1_200",
            data_path=Path(h_path),
            n=PASS1_N,
        ),
        MiniJob(
            surface="h_hard",
            benchmark_id="h_hard",
            adaptor_key="c1_or1_200",
            data_path=Path(h_path),
            n=FIG4_N,
            bin_filter="B0",
        ),
        MiniJob(
            surface="math500",
            benchmark_id="math500",
            adaptor_key="c1_math500",
            data_path=Path(math500),
            n=FIG4_N,
        ),
        MiniJob(
            surface="aime_union",
            benchmark_id="aime_union",
            adaptor_key="c1_aime_union",
            data_path=Path(aime_union),
            n=FIG4_N,
        ),
        MiniJob(
            surface="amc23",
            benchmark_id="amc23",
            adaptor_key="c1_amc23",
            data_path=Path(amc23),
            n=PASS1_N,
        ),
        MiniJob(
            surface="hmmt25",
            benchmark_id="hmmt25",
            adaptor_key="c1_hmmt25",
            data_path=Path(hmmt25),
            n=PASS1_N,
        ),
    ]
    if include_h_k128:
        jobs.append(
            MiniJob(
                surface="h_k128",
                benchmark_id="h_k128",
                adaptor_key="c1_or1_200",
                data_path=Path(h_path),
                n=H_K,
            )
        )
    return jobs


def build_h_mini_n512_job(h_mini_path: Path) -> MiniJob:
    """Ch.3 gate set: one file, n=512, every sample kept for pass@k k=1..512."""
    return MiniJob(
        surface="h_mini",
        benchmark_id="h_mini",
        adaptor_key="c1_or1_200",
        data_path=Path(h_mini_path),
        n=H_MINI_N,
    )


def parse_extra_set_spec(spec: str) -> tuple[str, Path]:
    """Parse ``name=/path/to.jsonl`` for ``--extra-set``."""
    raw = str(spec).strip()
    if "=" not in raw:
        raise ValueError(
            f"--extra-set expects name=path, got {spec!r}"
        )
    name, path_s = raw.split("=", 1)
    name = name.strip()
    path_s = path_s.strip()
    if not name or not path_s:
        raise ValueError(f"--extra-set expects name=path, got {spec!r}")
    if any(ch in name for ch in "/\\"):
        raise ValueError(f"--extra-set name must be a surface id, got {name!r}")
    return name, Path(path_s)


def build_extra_jobs(
    specs: Sequence[str],
    *,
    n: int = H_MINI_N,
    adaptor_key: str = "c1_or1_200",
) -> list[MiniJob]:
    """Arbitrary problem jsonl jobs with the same H-mini rendering/scorer path."""
    jobs: list[MiniJob] = []
    seen: set[str] = set()
    for spec in specs:
        name, path = parse_extra_set_spec(spec)
        if name in seen:
            raise ValueError(f"duplicate --extra-set name {name!r}")
        seen.add(name)
        jobs.append(
            MiniJob(
                surface=name,
                benchmark_id=name,
                adaptor_key=str(adaptor_key),
                data_path=Path(path),
                n=int(n),
            )
        )
    return jobs


def plan_summary(jobs: Sequence[MiniJob], *, base_seed: int) -> dict:
    refuse_binning_seed(int(base_seed), context="full_protocol")
    if int(base_seed) in {0, 1}:
        raise ValueError(f"full protocol seed {base_seed} collides with eval seeds {{0,1}}")
    fig4 = [j.surface for j in jobs if j.n == FIG4_N]
    pass1_only = [j.surface for j in jobs if j.n == PASS1_N]
    return {
        "base_seed": int(base_seed),
        "n_seeds": 1,
        "benchmarks_pass1": [
            "h",
            "h_hard",
            "math500",
            "amc23",
            "aime24",
            "aime25",
            "aime26",
            "hmmt25",
        ],
        "aime_sampled_as": "aime_union",
        "fig4_n512": fig4,
        "pass1_sampled_at_16": pass1_only,
        "pass1_from_n512": fig4,
        "jobs": [
            {
                "surface": j.surface,
                "n": j.n,
                "bin_filter": j.bin_filter,
                "data_path": str(j.data_path),
            }
            for j in jobs
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    from opd_eval.mini_protocol import (
        DEFAULT_H_MINI_PATH,
        DEFAULT_MATH500_MINI_PATH,
        build_parser,
        worker_main,
    )

    parser = build_parser()
    parser.add_argument(
        "--h-k128",
        action="store_true",
        help="Also draw H at K=128 (C00, C00', C01, C01g, C01f)",
    )
    parser.add_argument(
        "--only-h-mini-n512",
        action="store_true",
        help="Run only --h-mini at n=512. Samples stay in samples.jsonl "
        "so pass@k for k=1..512 is unbiased.",
    )
    parser.add_argument(
        "--extra-set",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Evaluate an arbitrary problem jsonl with the H-mini path "
        "(same C.1 prompt, scorer, sampler, seed, sharding/resume, output). "
        "Repeatable. Example: --extra-set h_b0_extra=data/h400_b0_extra50.jsonl",
    )
    parser.add_argument(
        "--extra-n",
        type=int,
        default=H_MINI_N,
        help=f"Samples per problem for each --extra-set (default {H_MINI_N})",
    )
    parser.add_argument(
        "--only-extra",
        action="store_true",
        help="Run only --extra-set jobs (skip the fixed full / H-mini surfaces).",
    )
    args = parser.parse_args(argv)
    if args.worker:
        return worker_main(args)
    h_path = Path(args.h_mini)
    if h_path == DEFAULT_H_MINI_PATH:
        h_path = DEFAULT_H
    math500 = Path(args.math500_mini)
    if math500 == DEFAULT_MATH500_MINI_PATH:
        math500 = DEFAULT_MATH500
    if args.only_extra:
        if not args.extra_set:
            raise SystemExit("--only-extra requires at least one --extra-set")
        jobs = []
    elif args.only_h_mini_n512:
        jobs = [build_h_mini_n512_job(Path(args.h_mini))]
    else:
        jobs = build_full_jobs(
            h_path=h_path,
            math500=math500,
            aime_union=args.aime_union,
            amc23=args.amc23,
            hmmt25=args.hmmt25,
            include_h_k128=bool(args.h_k128),
        )
    extra_jobs = build_extra_jobs(args.extra_set, n=int(args.extra_n)) if args.extra_set else []
    jobs.extend(extra_jobs)
    # Point the H jobs at the v5 H file when the mini flag was left at its default.
    summary = plan_summary(jobs, base_seed=int(args.base_seed))
    if args.only_h_mini_n512 and not args.only_extra:
        summary["mode"] = "h_mini_n512"
        summary["samples_kept"] = "samples.jsonl"
        summary["pass_at_k"] = "1..512"
    if args.only_extra:
        summary["mode"] = "extra_only"
    if extra_jobs:
        summary["extra_sets"] = [
            {"surface": j.surface, "n": j.n, "data_path": str(j.data_path)}
            for j in extra_jobs
        ]
    if args.dry_plan:
        print(json.dumps(summary, indent=2))
        return 0

    run_total = _plan_run_total(jobs, int(args.base_seed))
    started = time.time()
    write_progress(
        args.out_root,
        benchmark="starting",
        n_done=count_run_samples(args.out_root),
        n_total=run_total,
        started_at=started,
        status="running",
    )
    n_gpus = int(args.n_gpus)
    if args.gpu_ids:
        n_gpus = len([x for x in args.gpu_ids.split(",") if x.strip()])
    print(
        "EFFECTIVE_CONFIG "
        + json.dumps(
            {
                "model_dir": str(args.model_dir),
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "max_new_tokens": args.max_new_tokens,
                "max_model_len": args.max_model_len,
                "base_seed": int(args.base_seed),
                "n_gpus": n_gpus,
                "gpu_ids": args.gpu_ids,
                "max_num_seqs": int(args.max_num_seqs),
                "prompt": "C.1 + <think>\\n",
                "jobs": summary.get("jobs"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    reports = []
    for job in jobs:
        reports.append(
            run_job(
                job,
                out_root=args.out_root,
                model_dir=args.model_dir,
                base_seed=int(args.base_seed),
                dry_plan=False,
                use_vllm=True,
                n_gpus=n_gpus,
                max_num_seqs=int(args.max_num_seqs),
                generate_batch_size=int(args.generate_batch_size),
                max_new_tokens=int(args.max_new_tokens),
                max_model_len=int(args.max_model_len),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                gpu_memory_utilization=float(args.gpu_memory_utilization),
                gpu_ids=args.gpu_ids,
                run_started_at=started,
                run_n_total=run_total,
            )
        )
    write_progress(
        args.out_root,
        benchmark="done",
        n_done=count_run_samples(args.out_root),
        n_total=run_total,
        started_at=started,
        status="done",
    )
    print(json.dumps({"plan": summary, "jobs": reports}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
