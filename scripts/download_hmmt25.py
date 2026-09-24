#!/usr/bin/env python3
"""Download MathArena/hmmt_feb_2025 → data/hmmt25_bench_schema.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def download(out_path: Path) -> int:
    from datasets import load_dataset

    ds = load_dataset("MathArena/hmmt_feb_2025", split="train")
    rows = []
    for row in ds:
        rows.append(
            {
                "question": row["problem"],
                "ground_truth": row["answer"],
                "unique_id": int(row["problem_idx"]) - 1,
                "problem_idx": int(row["problem_idx"]),
                "problem_type": row["problem_type"],
            }
        )
    if len(rows) != 30:
        raise SystemExit(f"expected 30 HMMT problems, got {len(rows)}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {out_path} n={len(rows)}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "hmmt25_bench_schema.jsonl",
    )
    args = p.parse_args()
    return download(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
