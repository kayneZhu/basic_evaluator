"""G1: one student on its C.2 H-mini, eval protocol, every bin.

n=16, T=0.6, top_p=0.95, max_tokens=10240, seed 20261001,
``OPD_PROMPT_FORMAT=c2_nothink``. A one-problem smoke runs first.
Samples carry correct / length / terminal id / truncated / think-tag
literals / 50-gram x3 repetition, same scoring path as mini-protocol.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from adaptors.prompt_format import render_c2_prompt, resolve_prompt_format
from opd_eval.length import resolve_eval_max_model_len
from opd_eval.mini_protocol import MiniJob, run_job, sampling_params_kwargs
from opd_eval.stats import unbiased_pass_at_k

SPEC_N = 16
SPEC_TEMPERATURE = 0.6
SPEC_TOP_P = 0.95
SPEC_MAX_NEW_TOKENS = 10240
SPEC_SEED = 20261001
SMOKE_MAX_NEW_TOKENS = 64


def prompt_id_report(
    problems: Sequence[str],
    *,
    train_render,
    eval_render,
    encode,
) -> Dict[str, Any]:
    """Train and eval C.2 strings, and their token ids, must match."""
    n_bad_str = 0
    n_bad_ids = 0
    first_ids: Optional[list[int]] = None
    lengths: list[int] = []
    for problem in problems:
        train_text = train_render(problem)
        eval_text = eval_render(problem)
        train_ids = [int(x) for x in encode(train_text)]
        eval_ids = [int(x) for x in encode(eval_text)]
        if train_text != eval_text:
            n_bad_str += 1
        if train_ids != eval_ids:
            n_bad_ids += 1
        if first_ids is None:
            first_ids = eval_ids
        lengths.append(len(eval_ids))
    return {
        "n": len(problems),
        "n_string_mismatch": n_bad_str,
        "n_id_mismatch": n_bad_ids,
        "equal": n_bad_str == 0 and n_bad_ids == 0 and len(problems) > 0,
        "first_n_ids": len(first_ids or []),
        "max_prompt_ids": max(lengths) if lengths else 0,
        "min_prompt_ids": min(lengths) if lengths else 0,
    }


def g1_table(
    samples: Sequence[Mapping[str, Any]],
    bin_of: Mapping[str, str],
    *,
    n: int,
) -> Dict[str, Any]:
    """Per-bin pass@1, pass@n, trunc, terminals, think-tag, repetition, median len."""
    by_problem: Dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in samples:
        by_problem[str(row["problem_id"])].append(row)
    bins: Dict[str, list[str]] = defaultdict(list)
    for pid in by_problem:
        if pid not in bin_of:
            raise ValueError(f"{pid} has no bin")
        bins[str(bin_of[pid])].append(pid)

    def _one(pids: Sequence[str]) -> Dict[str, Any]:
        rows = [r for pid in pids for r in by_problem[pid]]
        if not rows:
            raise ValueError("empty bin")
        pass1 = []
        passn = []
        for pid in pids:
            group = by_problem[pid]
            if len(group) != int(n):
                raise ValueError(f"{pid} has {len(group)} samples, expected {n}")
            idxs = [int(r["sample_idx"]) for r in group]
            if sorted(idxs) != list(range(int(n))):
                raise ValueError(f"{pid} sample_idx not 0..{n - 1}")
            ordered = sorted(group, key=lambda r: int(r["sample_idx"]))
            c = sum(1 for r in ordered if bool(r["verified"]))
            pass1.append(unbiased_pass_at_k(int(n), c, 1))
            passn.append(unbiased_pass_at_k(int(n), c, int(n)))
        think = 0
        rep = 0
        trunc = 0
        lengths = []
        terminals: Counter[str] = Counter()
        for row in rows:
            if "repetition" not in row:
                raise ValueError("sample missing repetition flag")
            open_tag = bool(row.get("response_contains_think_open"))
            close_tag = bool(row.get("response_contains_think_close"))
            if open_tag or close_tag:
                think += 1
            if bool(row["repetition"]):
                rep += 1
            if bool(row["truncated"]):
                trunc += 1
            lengths.append(int(row["n_tokens"]))
            terminals[str(row.get("terminal_token_id"))] += 1
        denom = len(rows)
        return {
            "n_problems": len(pids),
            "n_samples": denom,
            "pass@1": sum(pass1) / len(pass1),
            f"pass@{int(n)}": sum(passn) / len(passn),
            "trunc_rate": trunc / denom,
            "think_tag_rate": think / denom,
            "repetition_rate": rep / denom,
            "median_len": statistics.median(lengths),
            "terminal_ids": dict(terminals),
        }

    order = ["B0", "B1", "B2", "B3"]
    per_bin = {name: _one(bins[name]) for name in order if bins.get(name)}
    pooled_ids = [pid for name in order for pid in bins.get(name, [])]
    return {"n": int(n), "per_bin": per_bin, "pooled": _one(pooled_ids)}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"refuse overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _load_train_render():
    root = os.environ.get("OPD_TRAIN_ROOT", "")
    if root and root not in sys.path:
        sys.path.insert(0, root)
    from opd_frontier.data.prompt import render_c2_prompt as train_render

    return train_render


def _run_one(
    *,
    data_path: Path,
    out_root: Path,
    benchmark_id: str,
    n: int,
    model_dir: Path,
    max_new_tokens: int,
    max_model_len: int,
    n_gpus: int,
    max_num_seqs: int,
    generate_batch_size: int,
    gpu_memory_utilization: float,
) -> Dict[str, Any]:
    job = MiniJob(
        surface=benchmark_id,
        benchmark_id=benchmark_id,
        adaptor_key="c1_or1_200",
        data_path=data_path,
        n=int(n),
    )
    return run_job(
        job,
        out_root=out_root,
        model_dir=model_dir,
        base_seed=SPEC_SEED,
        use_vllm=True,
        n_gpus=int(n_gpus),
        max_num_seqs=int(max_num_seqs),
        generate_batch_size=int(generate_batch_size),
        max_new_tokens=int(max_new_tokens),
        max_model_len=int(max_model_len),
        temperature=SPEC_TEMPERATURE,
        top_p=SPEC_TOP_P,
        gpu_memory_utilization=float(gpu_memory_utilization),
    )


def _assert_smoke(out_root: Path, question: str) -> None:
    bench = out_root / "smoke"
    samples = _read_jsonl(bench / "samples.jsonl")
    if len(samples) < 1:
        raise RuntimeError("smoke produced no samples")
    row = samples[0]
    for key in ("verified", "n_tokens", "terminal_token_id", "truncated", "repetition"):
        if key not in row:
            raise RuntimeError(f"smoke sample missing {key}")
    work_files = sorted(bench.glob("_gpu*_work.json"))
    if not work_files:
        raise RuntimeError("smoke did not record its work list")
    work = json.loads(work_files[0].read_text(encoding="utf-8"))
    prompt = str(work[0]["prompt"])
    expected = render_c2_prompt(question)
    if prompt != expected:
        raise RuntimeError("smoke prompt is not the C.2 eval string")
    if not prompt.endswith("<think>\n\n</think>\n\n"):
        raise RuntimeError("smoke prompt lost the closed think block")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h-mini", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=SPEC_N)
    parser.add_argument("--temperature", type=float, default=SPEC_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=SPEC_TOP_P)
    parser.add_argument("--max-new-tokens", type=int, default=SPEC_MAX_NEW_TOKENS)
    parser.add_argument("--base-seed", type=int, default=SPEC_SEED)
    parser.add_argument("--n-gpus", type=int, default=5)
    parser.add_argument("--max-num-seqs", type=int, default=12)
    parser.add_argument("--generate-batch-size", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args(argv)

    if resolve_prompt_format(required=True) != "c2_nothink":
        raise SystemExit("G1 requires OPD_PROMPT_FORMAT=c2_nothink")
    if (
        int(args.n) != SPEC_N
        or float(args.temperature) != SPEC_TEMPERATURE
        or float(args.top_p) != SPEC_TOP_P
        or int(args.max_new_tokens) != SPEC_MAX_NEW_TOKENS
        or int(args.base_seed) != SPEC_SEED
    ):
        raise SystemExit(
            "G1 params must stay n=16 T=0.6 top_p=0.95 max_tokens=10240 seed=20261001"
        )
    os.environ["OPD_PROMPT_FORMAT"] = "c2_nothink"
    os.environ["OPD_PROMPT_FORMAT_REQUIRED"] = "1"

    problems = _read_jsonl(Path(args.h_mini))
    if len(problems) != 200:
        raise SystemExit(f"H-mini has {len(problems)} rows, expected 200")
    questions = [str(r["question"]) for r in problems]
    bin_of = {str(r.get("or1_id") or r["problem_id"]): str(r["bin"]) for r in problems}

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), trust_remote_code=True)
    report = prompt_id_report(
        questions,
        train_render=_load_train_render(),
        eval_render=render_c2_prompt,
        encode=lambda text: tokenizer.encode(text, add_special_tokens=False),
    )
    print("PROMPT_IDS " + json.dumps(report), flush=True)
    if not report["equal"]:
        raise SystemExit("eval prompt ids differ from training ids")

    max_model_len = resolve_eval_max_model_len(SPEC_MAX_NEW_TOKENS)
    prompt_over = int(report["max_prompt_ids"]) > 1024
    if prompt_over:
        max_model_len = int(report["max_prompt_ids"]) + SPEC_MAX_NEW_TOKENS
    full_sampling = sampling_params_kwargs(
        SPEC_SEED,
        temperature=SPEC_TEMPERATURE,
        top_p=SPEC_TOP_P,
        max_tokens=SPEC_MAX_NEW_TOKENS,
    )
    params = {
        "model": str(args.model_dir),
        "prompt_format": "c2_nothink",
        "n_problems": len(problems),
        "n": SPEC_N,
        "temperature": SPEC_TEMPERATURE,
        "top_p": SPEC_TOP_P,
        "max_new_tokens": SPEC_MAX_NEW_TOKENS,
        "max_model_len": max_model_len,
        "prompt_over_1024": prompt_over,
        "base_seed": SPEC_SEED,
        "n_gpus": int(args.n_gpus),
        "max_num_seqs": int(args.max_num_seqs),
        "generate_batch_size": int(args.generate_batch_size),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "sampling": full_sampling,
        "h_mini": str(args.h_mini),
        "out_root": str(args.out_root),
    }
    print("G1_CONFIG " + json.dumps(params), flush=True)
    if int(full_sampling["max_tokens"]) != SPEC_MAX_NEW_TOKENS:
        raise SystemExit("sampling max_tokens is not 10240")
    if float(full_sampling["temperature"]) != SPEC_TEMPERATURE:
        raise SystemExit("sampling temperature is not 0.6")
    if float(full_sampling["top_p"]) != SPEC_TOP_P:
        raise SystemExit("sampling top_p is not 0.95")

    out_root = Path(args.out_root)
    raw = str(out_root)
    if raw == "/root/autodl-fs" or raw.startswith("/root/autodl-fs/"):
        raise SystemExit(f"refusing {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_smoke:
        smoke_path = out_root / "smoke_problems.jsonl"
        if not smoke_path.exists():
            _write_new(smoke_path, json.dumps(problems[0], ensure_ascii=False) + "\n")
        smoke_len = resolve_eval_max_model_len(SMOKE_MAX_NEW_TOKENS)
        smoke_sampling = sampling_params_kwargs(
            SPEC_SEED,
            temperature=SPEC_TEMPERATURE,
            top_p=SPEC_TOP_P,
            max_tokens=SMOKE_MAX_NEW_TOKENS,
        )
        print(
            "SMOKE_CONFIG "
            + json.dumps(
                {
                    "n_problems": 1,
                    "n": 1,
                    "n_gpus": 1,
                    "max_new_tokens": SMOKE_MAX_NEW_TOKENS,
                    "max_model_len": smoke_len,
                    "temperature": SPEC_TEMPERATURE,
                    "top_p": SPEC_TOP_P,
                    "prompt_format": "c2_nothink",
                    "sampling": smoke_sampling,
                }
            ),
            flush=True,
        )
        _run_one(
            data_path=smoke_path,
            out_root=out_root,
            benchmark_id="smoke",
            n=1,
            model_dir=Path(args.model_dir),
            max_new_tokens=SMOKE_MAX_NEW_TOKENS,
            max_model_len=smoke_len,
            n_gpus=1,
            max_num_seqs=1,
            generate_batch_size=1,
            gpu_memory_utilization=float(args.gpu_memory_utilization),
        )
        _assert_smoke(out_root, questions[0])
        print("SMOKE_OK", flush=True)

    _run_one(
        data_path=Path(args.h_mini),
        out_root=out_root,
        benchmark_id="hmini",
        n=SPEC_N,
        model_dir=Path(args.model_dir),
        max_new_tokens=SPEC_MAX_NEW_TOKENS,
        max_model_len=max_model_len,
        n_gpus=int(args.n_gpus),
        max_num_seqs=int(args.max_num_seqs),
        generate_batch_size=int(args.generate_batch_size),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
    )
    samples = _read_jsonl(out_root / "hmini" / "samples.jsonl")
    table = g1_table(samples, bin_of, n=SPEC_N)
    table_path = out_root / "g1_table.json"
    payload = json.dumps(table, indent=2) + "\n"
    if table_path.exists():
        table_path.write_text(payload, encoding="utf-8")
    else:
        _write_new(table_path, payload)
    print("G1_TABLE " + json.dumps(table), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
