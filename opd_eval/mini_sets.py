"""Build deterministic mini eval sets (parametrized; do not materialize H-mini by default).

H-mini: stratified from H \\ Hs (per-bin size is an owner parameter).
H-hard-mini: 80 from B0 of H \\ Hs, with H-mini's B0 problems as its prefix.
MATH500-mini: 200 problems, selection seed 20260924 (binning seed is OK here —
this is a *subset draw*, not an evaluation sample seed).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

BIN_NAMES = ("B0", "B1", "B2", "B3")
MATH500_MINI_SEED = 20260924  # selection seed for MATH500-mini only
MATH500_MINI_SIZE = 200
H_HARD_MINI_SIZE = 80


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def problem_id_of(row: Mapping[str, Any]) -> str:
    if row.get("or1_id"):
        return str(row["or1_id"])
    if row.get("problem_id"):
        return str(row["problem_id"])
    if row.get("unique_id") is not None:
        return str(row["unique_id"])
    raise KeyError(f"row has no problem id keys: {list(row)[:8]}")


def h_minus_hs(
    h_rows: Sequence[Mapping[str, Any]],
    hs_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    hs_ids = {problem_id_of(r) for r in hs_rows}
    out = [dict(r) for r in h_rows if problem_id_of(r) not in hs_ids]
    return out


def _bucket_by_bin(rows: Sequence[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {b: [] for b in BIN_NAMES}
    for r in rows:
        b = str(r.get("bin", ""))
        if b not in buckets:
            raise ValueError(f"unknown bin {b!r}")
        buckets[b].append(dict(r))
    for b in BIN_NAMES:
        buckets[b].sort(key=problem_id_of)
    return buckets


def seeded_take(
    rows: Sequence[Mapping[str, Any]],
    n: int,
    *,
    seed: int,
    salt: str,
) -> List[Dict[str, Any]]:
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if n > len(rows):
        raise ValueError(f"need {n} rows, have {len(rows)} (salt={salt})")
    rng = random.Random(f"{seed}|{salt}")
    order = list(range(len(rows)))
    rng.shuffle(order)
    return [dict(rows[i]) for i in order[:n]]


def build_h_mini(
    h_rows: Sequence[Mapping[str, Any]],
    hs_rows: Sequence[Mapping[str, Any]],
    *,
    per_bin: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Stratified H-mini from H\\Hs. ``per_bin`` is owner-chosen (e.g. 20 or 25)."""
    if per_bin <= 0:
        raise ValueError(f"per_bin must be > 0, got {per_bin}")
    rem = h_minus_hs(h_rows, hs_rows)
    buckets = _bucket_by_bin(rem)
    selected: List[Dict[str, Any]] = []
    for b in BIN_NAMES:
        selected.extend(
            seeded_take(buckets[b], per_bin, seed=seed, salt=f"h_mini|{b}")
        )
    return selected


def build_h_hard_mini(
    h_rows: Sequence[Mapping[str, Any]],
    hs_rows: Sequence[Mapping[str, Any]],
    h_mini: Sequence[Mapping[str, Any]],
    *,
    size: int = H_HARD_MINI_SIZE,
    seed: int,
) -> List[Dict[str, Any]]:
    """80 from B0 of H\\Hs; H-mini's B0 problems are a prefix of this set."""
    rem = h_minus_hs(h_rows, hs_rows)
    b0 = _bucket_by_bin(rem)["B0"]
    prefix = [dict(r) for r in h_mini if str(r.get("bin")) == "B0"]
    prefix_ids = {problem_id_of(r) for r in prefix}
    if not prefix_ids.issubset({problem_id_of(r) for r in b0}):
        raise ValueError("H-mini B0 is not a subset of H\\Hs B0")
    if len(prefix) > size:
        raise ValueError(f"H-mini B0 has {len(prefix)} > hard-mini size {size}")
    rest_pool = [r for r in b0 if problem_id_of(r) not in prefix_ids]
    need = size - len(prefix)
    rest = seeded_take(rest_pool, need, seed=seed, salt="h_hard_mini|B0")
    out = prefix + rest
    assert len(out) == size
    assert [problem_id_of(r) for r in out[: len(prefix)]] == [
        problem_id_of(r) for r in prefix
    ]
    return out


def build_math500_mini(
    rows: Sequence[Mapping[str, Any]],
    *,
    size: int = MATH500_MINI_SIZE,
    seed: int = MATH500_MINI_SEED,
) -> List[Dict[str, Any]]:
    return seeded_take(rows, size, seed=seed, salt="math500_mini")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--heldout-h", type=Path, required=True)
    p.add_argument("--hs", type=Path, required=True)
    p.add_argument("--math500", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--h-mini-per-bin",
        type=int,
        default=None,
        help="If set, write h_mini.jsonl (owner chooses 20 or 25). "
        "Omit to skip materializing H-mini.",
    )
    p.add_argument("--h-mini-seed", type=int, default=20260924)
    p.add_argument("--h-hard-mini-size", type=int, default=H_HARD_MINI_SIZE)
    p.add_argument("--write-h-hard-mini", action="store_true")
    p.add_argument("--write-math500-mini", action="store_true")
    p.add_argument("--math500-mini-size", type=int, default=MATH500_MINI_SIZE)
    p.add_argument("--math500-mini-seed", type=int, default=MATH500_MINI_SEED)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    h_rows = read_jsonl(args.heldout_h)
    hs_rows = read_jsonl(args.hs)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    h_mini: List[Dict[str, Any]] = []
    if args.h_mini_per_bin is not None:
        h_mini = build_h_mini(
            h_rows,
            hs_rows,
            per_bin=int(args.h_mini_per_bin),
            seed=int(args.h_mini_seed),
        )
        write_jsonl(out_dir / "h_mini.jsonl", h_mini)
        print(f"wrote {out_dir / 'h_mini.jsonl'} n={len(h_mini)}")

    if args.write_h_hard_mini:
        if not h_mini:
            raise SystemExit(
                "H-hard-mini needs H-mini B0 as prefix; pass --h-mini-per-bin"
            )
        hard = build_h_hard_mini(
            h_rows,
            hs_rows,
            h_mini,
            size=int(args.h_hard_mini_size),
            seed=int(args.h_mini_seed),
        )
        write_jsonl(out_dir / "h_hard_mini.jsonl", hard)
        print(f"wrote {out_dir / 'h_hard_mini.jsonl'} n={len(hard)}")

    if args.write_math500_mini:
        if args.math500 is None:
            raise SystemExit("--math500 is required for --write-math500-mini")
        m500 = build_math500_mini(
            read_jsonl(args.math500),
            size=int(args.math500_mini_size),
            seed=int(args.math500_mini_seed),
        )
        write_jsonl(out_dir / "math500_mini.jsonl", m500)
        print(f"wrote {out_dir / 'math500_mini.jsonl'} n={len(m500)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
