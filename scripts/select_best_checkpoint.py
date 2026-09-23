#!/usr/bin/env python3
"""
Select best + final checkpoints from mid-run validation metrics.

Reads ``metrics.jsonl`` (INTERFACE.md §5), keeps rows on the held-out
surface ``val_heldout_h``, restricts to saved-checkpoint steps (default
every 100), and writes ``selection.json`` reporting best and final
together. Primary metric defaults to pass@8.

Usage:
    python scripts/select_best_checkpoint.py \\
        --metrics path/to/metrics.jsonl \\
        --out path/to/selection.json

    python scripts/select_best_checkpoint.py \\
        --outputs-root ./outputs \\
        --run-id C01 \\
        --student-slug qwen3-1.7b-base
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parents[1]
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from opd_eval.validation import (
    PRIMARY_METRIC,
    SELECTION_SURFACE,
    load_metrics_jsonl,
    metrics_jsonl_path,
    select_best_and_final,
    selection_json_path,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics", type=Path, help="Path to metrics.jsonl")
    p.add_argument("--out", type=Path, help="Path to write selection.json")
    p.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    p.add_argument("--run-id", type=str)
    p.add_argument("--student-slug", type=str)
    p.add_argument(
        "--primary-metric",
        default=PRIMARY_METRIC,
        choices=("pass_at_8", "pass_at_1", "avg_at_8"),
    )
    p.add_argument("--selection-surface", default=SELECTION_SURFACE)
    p.add_argument(
        "--saved-steps",
        type=str,
        default=None,
        help="Comma-separated saved steps; default = every 100 present in file",
    )
    p.add_argument(
        "--final-step",
        type=int,
        default=None,
        help="Override final step (must be a saved step with a metrics row)",
    )
    args = p.parse_args(argv)

    if args.metrics is None:
        if not args.run_id or not args.student_slug:
            p.error("provide --metrics, or both --run-id and --student-slug")
        metrics_path = metrics_jsonl_path(
            args.outputs_root, args.run_id, args.student_slug
        )
    else:
        metrics_path = args.metrics

    if args.out is None:
        if args.run_id and args.student_slug:
            out_path = selection_json_path(
                args.outputs_root, args.run_id, args.student_slug
            )
        else:
            out_path = metrics_path.with_name("selection.json")
    else:
        out_path = args.out

    rows = load_metrics_jsonl(metrics_path)
    saved_steps = None
    if args.saved_steps:
        saved_steps = [int(x.strip()) for x in args.saved_steps.split(",") if x.strip()]

    report = select_best_and_final(
        rows,
        saved_steps=saved_steps,
        final_step=args.final_step,
        primary_metric=args.primary_metric,
        selection_surface=args.selection_surface,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report.as_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"best_step={report.best.step} "
        f"{args.primary_metric}={report.best.metric(args.primary_metric):.6f}  "
        f"final_step={report.final.step} "
        f"{args.primary_metric}={report.final.metric(args.primary_metric):.6f}"
    )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
