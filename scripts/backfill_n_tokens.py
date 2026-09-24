#!/usr/bin/env python3
"""One-off: back-fill real n_tokens + finish_reason for old mini-eval samples.

Old rows stored n_tokens = len(response.split()) (words). This script retokenizes
``response`` with the model tokenizer and rewrites:

  n_tokens      = len(tokenizer.encode(response))
  n_words       = previous word-count field (preserved / set from split)
  finish_reason = "length" iff n_tokens == max_new_tokens and no EOS/stop id
                  else "stop"

Does NOT re-generate. Safe to run on a copy first. Optional --limit for smoke.

Example (smoke)::

    python scripts/backfill_n_tokens.py \\
      --samples /path/to/_gpu0_samples.jsonl \\
      --tokenizer /root/autodl-tmp/models/Qwen/Qwen3-1.7B-Base \\
      --limit 20 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

# Allow `python scripts/backfill_n_tokens.py` from eval root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opd_eval.contract import vllm_stop_token_ids
from opd_eval.length import STUDENT_MAX_NEW_TOKENS


def _infer_finish_reason(
    token_ids: Sequence[int],
    *,
    max_new_tokens: int,
    stop_ids: Set[int],
) -> str:
    n = len(token_ids)
    has_eos = any(int(t) in stop_ids for t in token_ids)
    if n == int(max_new_tokens) and not has_eos:
        return "length"
    return "stop"


def backfill_file(
    path: Path,
    tokenizer: Any,
    *,
    max_new_tokens: int,
    stop_ids: Set[int],
    limit: Optional[int] = None,
    dry_run: bool = False,
) -> Dict[str, int]:
    rows: List[Dict[str, Any]] = []
    n_changed = 0
    n_skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= int(limit):
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = str(rec.get("response", ""))
            # Prefer encode without specials — completion suffix only.
            ids = tokenizer.encode(text, add_special_tokens=False)
            n_tok = len(ids)
            old_n = rec.get("n_tokens")
            # Preserve prior word-count if it looked like the old field.
            if "n_words" not in rec:
                rec["n_words"] = (
                    int(old_n)
                    if isinstance(old_n, int) and old_n != n_tok
                    else len(text.split())
                )
            reason = _infer_finish_reason(
                ids, max_new_tokens=max_new_tokens, stop_ids=stop_ids
            )
            if rec.get("n_tokens") == n_tok and rec.get("finish_reason") == reason:
                n_skipped += 1
            else:
                n_changed += 1
            rec["n_tokens"] = n_tok
            rec["finish_reason"] = reason
            rows.append(rec)

    if dry_run or limit is not None:
        # --limit never rewrites (would truncate); dry-run never writes.
        print(
            json.dumps(
                {
                    "path": str(path),
                    "n_read": len(rows),
                    "n_would_change": n_changed,
                    "n_already_ok": n_skipped,
                    "dry_run": bool(dry_run) or limit is not None,
                    "sample0": {
                        "n_tokens": rows[0].get("n_tokens"),
                        "n_words": rows[0].get("n_words"),
                        "finish_reason": rows[0].get("finish_reason"),
                    }
                    if rows
                    else None,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return {"n_read": len(rows), "n_changed": n_changed, "n_skipped": n_skipped}

    tmp = path.with_suffix(path.suffix + ".backfill.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    return {"n_read": len(rows), "n_changed": n_changed, "n_skipped": n_skipped}


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--samples",
        type=Path,
        required=True,
        help="One jsonl file (or use --glob via shell)",
    )
    p.add_argument(
        "--tokenizer",
        type=Path,
        required=True,
        help="HF model dir used for encode()",
    )
    p.add_argument("--max-new-tokens", type=int, default=STUDENT_MAX_NEW_TOKENS)
    p.add_argument("--limit", type=int, default=None, help="Only first N rows")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(args.tokenizer), trust_remote_code=True)
    stops = set(int(x) for x in vllm_stop_token_ids())
    stats = backfill_file(
        args.samples,
        tok,
        max_new_tokens=int(args.max_new_tokens),
        stop_ids=stops,
        limit=args.limit,
        dry_run=bool(args.dry_run),
    )
    if not args.dry_run:
        print(json.dumps({"path": str(args.samples), **stats}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
