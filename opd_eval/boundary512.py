"""n=512 C.2 boundary eval on AIME 2024/2025/2026 and AMC 2023.

Same generation path as the n=256 H-mini boundary run: ``run_job`` in
``opd_eval.mini_protocol`` (per-sample seeds, resume on
``(problem_id, sample_idx)``, contract stop tokens). ``g1_hmini`` stays
locked at n=16; this entry is the large-k job.

Pre-registered sampler (not flags): T=0.6, top_p=0.95, max new tokens
10240, ``OPD_PROMPT_FORMAT=c2_nothink``, stops 151645 and 151643.
``--n`` and ``--seed`` are arguments so a CPU dry-run can use a tiny n;
the host wrapper passes n=512 and seed 20261005.

The on-disk AMC file is the text-only 40-problem set (AMC 12A/12B 2023
with 10 contest problems absent). The pre-registration asked for 50.
This runner evaluates whatever is on disk (40 or 50) and records the
gap. ``--strict-registered`` refuses to start unless AMC has 50 rows.

Chunks: each chunk is its own ``run_job`` directory. A finished chunk is
compacted to ``chunks/<id>.jsonl.gz`` and marked ``.done`` before the
next chunk starts, so a crash keeps completed chunks. The compact rows
match ``docs/eng/reports/data/c2_base/boundary_*.jsonl.gz``.

Usage (GPU host)::

    OPD_PROMPT_FORMAT=c2_nothink python -m opd_eval.boundary512 \\
        --model-dir /path/to/hf --out-dir /path/out \\
        --sets aime24,aime25,aime26,amc23 --n 512 --seed 20261005
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from adaptors.adaptor_factory import AdaptorFactory
from adaptors.prompt_format import (
    C2_ASSISTANT_PREFILL,
    render_c2_prompt,
    resolve_prompt_format,
)
from opd_eval.contract import vllm_stop_token_ids
from opd_eval.length import resolve_eval_max_model_len
from opd_eval.mini_protocol import MiniJob, _load_done_keys, run_job

EVAL_ROOT = Path(__file__).resolve().parents[1]

SPEC_TEMPERATURE = 0.6
SPEC_TOP_P = 0.95
SPEC_MAX_NEW_TOKENS = 10240
SPEC_SEED = 20261005
SPEC_N = 512
SPEC_STOPS = (151645, 151643)
# Qwen3 ids for the C.2 assistant prefill ``<think>\n\n</think>\n\n``.
C2_PREFILL_IDS = (151667, 271, 151668, 271)

# Year rows live in the union file. Standalone year files reuse unique_id
# 0–29, and the year adaptor would then drop AIME 2025/2026.
UNION_RELATIVE = "data/aime24_25_26_bench_schema.jsonl"
AMC_RELATIVE = "data/amc23_bench_schema.jsonl"

SET_ADAPTOR = {
    "aime24": "c1_aime24",
    "aime25": "c1_aime25",
    "aime26": "c1_aime26",
    "amc23": "c1_amc23",
}
SET_PATH = {
    "aime24": UNION_RELATIVE,
    "aime25": UNION_RELATIVE,
    "aime26": UNION_RELATIVE,
    "amc23": AMC_RELATIVE,
}
REGISTERED_COUNTS = {"aime24": 30, "aime25": 30, "aime26": 30, "amc23": 50}
AIME_COUNTS = {"aime24": 30, "aime25": 30, "aime26": 30}

GenerateFn = Callable[[str, int], str]


def prefill_boundary_ok(token_ids: Sequence[int]) -> bool:
    if len(token_ids) < 4:
        return False
    return tuple(int(x) for x in token_ids[-4:]) == C2_PREFILL_IDS


def amc_missing_labels(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    """Contest problems absent from an AMC 12A+12B 2023 url list."""
    found = set()
    for row in rows:
        url = str(row.get("url") or "")
        match = re.search(r"(12A|12B)_Problems/Problem_(\d+)", url)
        if match:
            found.add((match.group(1), int(match.group(2))))
    missing: List[str] = []
    if not found:
        return missing
    for contest in ("12A", "12B"):
        for number in range(1, 26):
            if (contest, number) not in found:
                missing.append(f"2023 AMC {contest} #{number}")
    return missing


def _refuse_shared_fs(path: Path) -> None:
    raw = str(path)
    if raw == "/root/autodl-fs" or raw.startswith("/root/autodl-fs/"):
        raise SystemExit(f"refusing {path}")


def _load_train_render():
    path = EVAL_ROOT.parent / "train" / "opd_frontier" / "data" / "prompt.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("_opd_train_c2_prompt_b512", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.render_c2_prompt


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _read_jsonl(path: Path) -> List[dict[str, Any]]:
    rows: List[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl_gz(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    """Atomic gzip jsonl. Returns md5 of the finished file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_row(record: Mapping[str, Any], bin_name: str) -> Dict[str, Any]:
    """One boundary-table row. ``correct`` is the grader bit."""
    if "verified" in record:
        correct = bool(record["verified"])
    elif "correct" in record:
        correct = bool(record["correct"])
    else:
        raise KeyError("sample has neither verified nor correct")
    if "terminal_token_id" in record:
        end_token = record["terminal_token_id"]
        if "repetition" not in record:
            raise KeyError("sample with terminal_token_id is missing repetition")
    else:
        end_token = record.get("end_token_id")
    return {
        "problem_id": str(record["problem_id"]),
        "bin": str(bin_name),
        "sample_idx": int(record["sample_idx"]),
        "correct": correct,
        "truncated": bool(record.get("truncated", False)),
        "n_tokens": int(record["n_tokens"]),
        "end_token_id": end_token,
        "repetition": bool(record.get("repetition", False)),
    }


def compact_samples(
    samples_path: Path,
    gz_path: Path,
    *,
    bin_name: str,
    n: int,
) -> List[Dict[str, Any]]:
    compact = [compact_row(row, bin_name) for row in _read_jsonl(samples_path)]
    compact.sort(key=lambda row: (row["problem_id"], row["sample_idx"]))
    seen = set()
    by_problem: Dict[str, List[int]] = {}
    for row in compact:
        key = (row["problem_id"], row["sample_idx"])
        if key in seen:
            raise RuntimeError(f"duplicate sample {key}")
        seen.add(key)
        by_problem.setdefault(row["problem_id"], []).append(row["sample_idx"])
    expected_idx = list(range(int(n)))
    for pid, idxs in by_problem.items():
        if idxs != expected_idx:
            raise RuntimeError(f"{pid} sample_idx {idxs} != 0..{n - 1}")
    write_jsonl_gz(gz_path, compact)
    return compact


def _load_set(
    name: str,
    *,
    eval_root: Path,
    max_problems: Optional[int],
) -> tuple[str, List[dict[str, Any]], List[str]]:
    if name not in SET_ADAPTOR:
        raise SystemExit(f"unknown set {name}; choose from {sorted(SET_ADAPTOR)}")
    path = eval_root / SET_PATH[name]
    if not path.is_file():
        raise SystemExit(f"missing problem file {path}")
    adaptor = AdaptorFactory.create_adaptor(
        SET_ADAPTOR[name], str(path), thinking_mode=False
    )
    rows = list(adaptor.data)
    if name in AIME_COUNTS and max_problems is None and len(rows) != AIME_COUNTS[name]:
        raise SystemExit(f"{name} has {len(rows)} problems, expected {AIME_COUNTS[name]}")
    missing: List[str] = []
    if name == "amc23" and max_problems is None:
        if len(rows) not in (40, 50):
            raise SystemExit(
                f"amc23 has {len(rows)} problems; expected 40 (on disk) or 50 (registered)"
            )
        if len(rows) != REGISTERED_COUNTS["amc23"]:
            missing = amc_missing_labels(rows)
    if max_problems is not None:
        if max_problems < 1:
            raise SystemExit("--max-problems-per-set must be >= 1")
        rows = rows[: int(max_problems)]
    return SET_ADAPTOR[name], rows, missing


def _assert_renders(rows: Sequence[Mapping[str, Any]]) -> None:
    train_render = _load_train_render()
    for row in rows:
        question = str(row.get("question") or "")
        if not question:
            raise SystemExit("problem row has an empty question")
        rendered = render_c2_prompt(question)
        if not rendered.endswith(C2_ASSISTANT_PREFILL):
            raise SystemExit("C.2 prompt lost the closed think block")
        if train_render is not None and train_render(question) != rendered:
            raise SystemExit("eval C.2 string differs from train render_c2_prompt")


def _assert_prefill_ids(model_dir: Path, rows: Sequence[Mapping[str, Any]]) -> int:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    longest = 0
    for row in rows:
        text = render_c2_prompt(str(row["question"]))
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not prefill_boundary_ok(ids):
            raise SystemExit(
                f"prefill token ids {ids[-4:]} != {list(C2_PREFILL_IDS)}"
            )
        longest = max(longest, len(ids))
    return longest


def _fingerprint(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    keys = (
        "n",
        "seed",
        "sets",
        "chunk_problems",
        "max_problems_per_set",
        "temperature",
        "top_p",
        "max_new_tokens",
        "prompt_format",
        "stop_token_ids",
        "model_dir",
        "on_disk_counts",
    )
    return {key: cfg[key] for key in keys}


def _check_resume_config(path: Path, cfg: Mapping[str, Any]) -> None:
    fresh = _fingerprint(cfg)
    if not path.is_file():
        path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        return
    old = json.loads(path.read_text(encoding="utf-8"))
    if _fingerprint(old) != fresh:
        raise SystemExit(f"refusing to resume {path}: protocol fields differ")


def _chunk_ranges(n_rows: int, chunk_problems: int) -> List[tuple[int, int]]:
    ranges = []
    start = 0
    while start < n_rows:
        end = min(n_rows, start + int(chunk_problems))
        ranges.append((start, end))
        start = end
    return ranges


def _run_chunk(
    *,
    out_dir: Path,
    set_name: str,
    adaptor_key: str,
    rows: Sequence[Mapping[str, Any]],
    chunk_index: int,
    n: int,
    seed: int,
    model_dir: Path,
    use_vllm: bool,
    generate_fn: Optional[GenerateFn],
    n_gpus: int,
    max_num_seqs: int,
    generate_batch_size: int,
    gpu_memory_utilization: float,
    max_model_len: int,
) -> Path:
    benchmark_id = f"{set_name}_c{chunk_index:03d}"
    problem_path = out_dir / "_problems" / f"{benchmark_id}.jsonl"
    gz_path = out_dir / "chunks" / f"{benchmark_id}.jsonl.gz"
    done_path = out_dir / "chunks" / f"{benchmark_id}.done"
    if done_path.is_file() and gz_path.is_file():
        print(f"CHUNK_SKIP {benchmark_id}", flush=True)
        return gz_path
    if not problem_path.is_file():
        _write_jsonl(problem_path, rows)
    reloaded = AdaptorFactory.create_adaptor(
        adaptor_key, str(problem_path), thinking_mode=False
    )
    if len(reloaded.data) != len(rows):
        raise RuntimeError(
            f"{benchmark_id} reloaded {len(reloaded.data)} rows, wrote {len(rows)}"
        )
    samples_path = out_dir / benchmark_id / "samples.jsonl"
    expected = len(rows) * int(n)
    have = len(_load_done_keys(samples_path))
    if have != expected:
        job = MiniJob(
            surface=benchmark_id,
            benchmark_id=benchmark_id,
            adaptor_key=adaptor_key,
            data_path=problem_path,
            n=int(n),
        )
        print(
            f"CHUNK_START {benchmark_id} problems={len(rows)} "
            f"have={have} expected={expected}",
            flush=True,
        )
        run_job(
            job,
            out_root=out_dir,
            model_dir=model_dir,
            base_seed=int(seed),
            generate_fn=generate_fn,
            use_vllm=bool(use_vllm),
            n_gpus=int(n_gpus),
            max_num_seqs=int(max_num_seqs),
            generate_batch_size=int(generate_batch_size),
            max_new_tokens=SPEC_MAX_NEW_TOKENS,
            max_model_len=int(max_model_len),
            temperature=SPEC_TEMPERATURE,
            top_p=SPEC_TOP_P,
            gpu_memory_utilization=float(gpu_memory_utilization),
        )
        have = len(_load_done_keys(samples_path))
    if have != expected:
        raise RuntimeError(f"{benchmark_id} has {have} samples, expected {expected}")
    compact_samples(samples_path, gz_path, bin_name=set_name, n=int(n))
    done_path.write_text(
        json.dumps(
            {
                "benchmark_id": benchmark_id,
                "set": set_name,
                "n_problems": len(rows),
                "n": int(n),
                "n_rows": expected,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"CHUNK_DONE {benchmark_id} rows={expected}", flush=True)
    return gz_path


def _read_gz(path: Path) -> List[dict[str, Any]]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def merge_boundary(out_dir: Path, chunk_gz: Sequence[Path]) -> Path:
    merged: List[dict[str, Any]] = []
    seen = set()
    for path in chunk_gz:
        for row in _read_gz(path):
            key = (row["problem_id"], int(row["sample_idx"]))
            if key in seen:
                raise RuntimeError(f"duplicate sample {key} across chunks")
            seen.add(key)
            merged.append(row)
    dest = out_dir / "boundary.jsonl.gz"
    digest = write_jsonl_gz(dest, merged)
    print(f"BOUNDARY {dest} rows={len(merged)} md5={digest}", flush=True)
    return dest


def run_boundary(
    *,
    out_dir: Path,
    sets: Sequence[str],
    n: int,
    seed: int,
    model_dir: Optional[Path] = None,
    chunk_problems: int = 10,
    max_problems_per_set: Optional[int] = None,
    n_gpus: int = 5,
    max_num_seqs: int = 12,
    generate_batch_size: int = 128,
    gpu_memory_utilization: float = 0.90,
    cpu_stub: bool = False,
    generate_fn: Optional[GenerateFn] = None,
    dry_plan: bool = False,
    strict_registered: bool = False,
    eval_root: Path = EVAL_ROOT,
    check_tokenizer: bool = True,
) -> Dict[str, Any]:
    """Run or resume one model. Returns the completion payload."""
    if resolve_prompt_format(required=True) != "c2_nothink":
        raise SystemExit("boundary512 requires OPD_PROMPT_FORMAT=c2_nothink")
    os.environ["OPD_PROMPT_FORMAT"] = "c2_nothink"
    os.environ["OPD_PROMPT_FORMAT_REQUIRED"] = "1"
    stops = tuple(vllm_stop_token_ids())
    if stops != SPEC_STOPS:
        raise SystemExit(f"stop tokens {stops} != {SPEC_STOPS}")
    if int(chunk_problems) < 1:
        raise SystemExit("--chunk-problems must be >= 1")
    _refuse_shared_fs(Path(out_dir))

    loaded: List[tuple[str, str, List[dict[str, Any]]]] = []
    missing_amc: List[str] = []
    for name in sets:
        adaptor_key, rows, missing = _load_set(
            name, eval_root=eval_root, max_problems=max_problems_per_set
        )
        if name == "amc23":
            missing_amc = missing
        loaded.append((name, adaptor_key, rows))
    all_rows = [row for _, _, rows in loaded for row in rows]
    _assert_renders(all_rows)
    n_problems = len(all_rows)
    note = {
        "registered_counts": REGISTERED_COUNTS,
        "on_disk_counts": {name: len(rows) for name, _, rows in loaded},
        "n_problems": n_problems,
        "amc_missing": missing_amc,
        "registered_total": sum(REGISTERED_COUNTS[name] for name in sets),
    }
    print("PROTOCOL_NOTE " + json.dumps(note), flush=True)
    if strict_registered and any(
        note["on_disk_counts"][name] != REGISTERED_COUNTS[name] for name in sets
    ):
        raise SystemExit(2)
    plan = {
        "sets": list(sets),
        "n": int(n),
        "seed": int(seed),
        "n_problems": n_problems,
        "n_samples": n_problems * int(n),
        "chunk_problems": int(chunk_problems),
        "temperature": SPEC_TEMPERATURE,
        "top_p": SPEC_TOP_P,
        "max_new_tokens": SPEC_MAX_NEW_TOKENS,
        "prompt_format": "c2_nothink",
        "stop_token_ids": list(SPEC_STOPS),
        "prefill_ids": list(C2_PREFILL_IDS),
        **note,
    }
    if dry_plan:
        print("DRY_PLAN " + json.dumps(plan), flush=True)
        return plan

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    use_vllm = generate_fn is None and not cpu_stub
    if cpu_stub and generate_fn is None:
        def generate_fn(prompt: str, seed_i: int) -> str:  # noqa: A001
            del prompt, seed_i
            return r"\boxed{0}"
    if use_vllm:
        if model_dir is None or not Path(model_dir).is_dir():
            raise SystemExit("--model-dir must be an HF checkpoint directory")
        model_dir = Path(model_dir)
    else:
        model_dir = Path(model_dir) if model_dir is not None else Path("cpu-stub")

    max_model_len = resolve_eval_max_model_len(SPEC_MAX_NEW_TOKENS)
    if use_vllm and check_tokenizer:
        longest = _assert_prefill_ids(model_dir, all_rows)
        if longest > 1024:
            max_model_len = int(longest) + SPEC_MAX_NEW_TOKENS
        print(
            f"PREFILL_IDS ok tail={list(C2_PREFILL_IDS)} max_prompt_ids={longest} "
            f"max_model_len={max_model_len}",
            flush=True,
        )

    cfg = {
        **plan,
        "model_dir": str(model_dir),
        "max_problems_per_set": max_problems_per_set,
        "max_model_len": max_model_len,
        "n_gpus": int(n_gpus),
        "max_num_seqs": int(max_num_seqs),
        "generate_batch_size": int(generate_batch_size),
    }
    _check_resume_config(out_dir / "config.json", cfg)

    chunk_gz: List[Path] = []
    for name, adaptor_key, rows in loaded:
        for chunk_index, (start, end) in enumerate(
            _chunk_ranges(len(rows), int(chunk_problems))
        ):
            chunk_gz.append(
                _run_chunk(
                    out_dir=out_dir,
                    set_name=name,
                    adaptor_key=adaptor_key,
                    rows=rows[start:end],
                    chunk_index=chunk_index,
                    n=int(n),
                    seed=int(seed),
                    model_dir=model_dir,
                    use_vllm=use_vllm,
                    generate_fn=generate_fn,
                    n_gpus=int(n_gpus),
                    max_num_seqs=int(max_num_seqs),
                    generate_batch_size=int(generate_batch_size),
                    gpu_memory_utilization=float(gpu_memory_utilization),
                    max_model_len=max_model_len,
                )
            )
    boundary = merge_boundary(out_dir, chunk_gz)
    n_rows = 0
    with gzip.open(boundary, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                n_rows += 1
    if n_rows != n_problems * int(n):
        raise RuntimeError(f"boundary rows {n_rows} != {n_problems * int(n)}")
    digest = hashlib.md5(boundary.read_bytes()).hexdigest()
    payload = {
        "n_rows": n_rows,
        "n_problems": n_problems,
        "n": int(n),
        "seed": int(seed),
        "sets": list(sets),
        "boundary_md5": digest,
        "amc_missing": missing_amc,
        "prompt_format": "c2_nothink",
        "temperature": SPEC_TEMPERATURE,
        "top_p": SPEC_TOP_P,
        "max_new_tokens": SPEC_MAX_NEW_TOKENS,
        "stop_token_ids": list(SPEC_STOPS),
    }
    complete = out_dir / "COMPLETE"
    tmp = complete.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, complete)
    print("COMPLETE " + json.dumps(payload), flush=True)
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--sets",
        default="aime24,aime25,aime26,amc23",
        help="Comma list. Default: aime24,aime25,aime26,amc23",
    )
    parser.add_argument("--n", type=int, default=SPEC_N)
    parser.add_argument("--seed", type=int, default=SPEC_SEED)
    parser.add_argument("--chunk-problems", type=int, default=10)
    parser.add_argument("--n-gpus", type=int, default=5)
    parser.add_argument("--max-num-seqs", type=int, default=12)
    parser.add_argument("--generate-batch-size", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument(
        "--max-problems-per-set",
        type=int,
        default=None,
        help="Keep only the first N problems of each set (CPU tests).",
    )
    parser.add_argument(
        "--cpu-stub",
        action="store_true",
        help="Score a fixed boxed answer. No GPU and no vLLM.",
    )
    parser.add_argument(
        "--dry-plan",
        action="store_true",
        help="Print counts and the AMC gap. Do not generate.",
    )
    parser.add_argument(
        "--strict-registered",
        action="store_true",
        help="Exit 2 unless every set has its pre-registered count (AMC 50).",
    )
    args = parser.parse_args(argv)
    names = [part.strip() for part in str(args.sets).split(",") if part.strip()]
    if not names:
        raise SystemExit("empty --sets")
    try:
        run_boundary(
            out_dir=args.out_dir,
            sets=names,
            n=int(args.n),
            seed=int(args.seed),
            model_dir=args.model_dir,
            chunk_problems=int(args.chunk_problems),
            max_problems_per_set=args.max_problems_per_set,
            n_gpus=int(args.n_gpus),
            max_num_seqs=int(args.max_num_seqs),
            generate_batch_size=int(args.generate_batch_size),
            gpu_memory_utilization=float(args.gpu_memory_utilization),
            cpu_stub=bool(args.cpu_stub),
            dry_plan=bool(args.dry_plan),
            strict_registered=bool(args.strict_registered),
        )
    except SystemExit as exc:
        code = exc.code
        if code in (None, 0):
            return 0
        if isinstance(code, int):
            return code
        print(str(code), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
