"""Mini-protocol eval driver: one HF model dir → (n,c) + metrics jsonl.

Protocol (resume-safe on (problem_id, sample_idx)):

- n=512: H-hard-mini, MATH500-mini, AIME union
- n=16: H-mini non-B0 (B0 reuses H-hard-mini samples), AMC'23, HMMT'25

CPU tests: pass ``generate_fn`` / ``batch_generate_fn``, or ``--dry-plan``.
Live GPU: ``python -m opd_eval.mini_protocol --model-dir …`` launches
data-parallel vLLM (TP=1, one engine per GPU, max_num_seqs=64) with
per-request ``sample_seed`` and many requests per ``llm.generate`` call.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from adaptors.adaptor_factory import AdaptorFactory
from adaptors.base_adaptor import BaseAdaptor
from opd_eval.contract import vllm_stop_token_ids
from opd_eval.length import MAX_MODEL_LEN, STUDENT_MAX_NEW_TOKENS
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
# Batched path: work items (prompt/seed/…) → response texts (same order).
BatchGenerateFn = Callable[[Sequence[Mapping[str, Any]]], List[str]]

# Canonical materialized mini sets (server). Local mirror: docs/eng/eval_mini/.
DEFAULT_EVAL_MINI_DIR = Path("/root/autodl-tmp/data/processed/eval_mini")
DEFAULT_H_MINI_PATH = DEFAULT_EVAL_MINI_DIR / "h_mini.jsonl"
DEFAULT_H_HARD_MINI_PATH = DEFAULT_EVAL_MINI_DIR / "h_hard_mini.jsonl"
DEFAULT_MATH500_MINI_PATH = DEFAULT_EVAL_MINI_DIR / "math500_mini.jsonl"

DEFAULT_MAX_NUM_SEQS = 64
DEFAULT_N_GPUS = 4
DEFAULT_GPU_MEMORY_UTILIZATION = 0.90


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
    h_hard_mini: Path = DEFAULT_H_HARD_MINI_PATH,
    h_mini: Path = DEFAULT_H_MINI_PATH,
    math500_mini: Path = DEFAULT_MATH500_MINI_PATH,
    aime_union: Path = Path("data/aime24_25_26_bench_schema.jsonl"),
    amc23: Path = Path("data/amc23_bench_schema.jsonl"),
    hmmt25: Path = Path("data/hmmt25_bench_schema.jsonl"),
) -> List[MiniJob]:
    return [
        MiniJob(
            surface="h_hard_mini",
            benchmark_id="h_hard_mini",
            adaptor_key="c1_or1_200",
            data_path=Path(h_hard_mini),
            n=512,
        ),
        MiniJob(
            surface="math500_mini",
            benchmark_id="math500_mini",
            adaptor_key="c1_math500",
            data_path=Path(math500_mini),
            n=512,
        ),
        MiniJob(
            surface="aime_union",
            benchmark_id="aime_union",
            adaptor_key="c1_aime_union",
            data_path=Path(aime_union),
            n=512,
        ),
        MiniJob(
            surface="h_mini",
            benchmark_id="h_mini",
            adaptor_key="c1_or1_200",
            data_path=Path(h_mini),
            n=16,
            reuse_from="h_hard_mini",
            bin_exclude="B0",
        ),
        MiniJob(
            surface="amc23",
            benchmark_id="amc23",
            adaptor_key="c1_amc23",
            data_path=Path(amc23),
            n=16,
        ),
        MiniJob(
            surface="hmmt25",
            benchmark_id="hmmt25",
            adaptor_key="c1_hmmt25",
            data_path=Path(hmmt25),
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


def _score_and_append(
    adaptor: BaseAdaptor,
    work: Sequence[Mapping[str, Any]],
    texts: Sequence[str],
    *,
    samples_path: Path,
    benchmark_id: str,
    flush_every: int = 64,
) -> int:
    if len(texts) != len(work):
        raise RuntimeError(
            f"batch generate returned {len(texts)} texts for {len(work)} requests"
        )
    batch: List[Dict[str, Any]] = []
    n_written = 0
    for w, text in zip(work, texts):
        scored = score_response(adaptor, text, w["ground_truth"])
        batch.append(
            {
                "problem_id": w["problem_id"],
                "sample_idx": w["sample_idx"],
                "seed": w["seed"],
                "benchmark_id": benchmark_id,
                "g": 0,
                **scored,
            }
        )
        if len(batch) >= flush_every:
            _append_jsonl(samples_path, batch)
            n_written += len(batch)
            batch.clear()
    if batch:
        _append_jsonl(samples_path, batch)
        n_written += len(batch)
    return n_written


def merge_shard_samples(bench_dir: Path, samples_path: Path) -> int:
    """Merge ``_gpu*_samples.jsonl`` into ``samples.jsonl`` (resume-safe)."""
    done = _load_done_keys(samples_path)
    added = 0
    for shard in sorted(bench_dir.glob("_gpu*_samples.jsonl")):
        batch: List[Dict[str, Any]] = []
        with open(shard, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                key = (str(rec["problem_id"]), int(rec["sample_idx"]))
                if key in done:
                    continue
                batch.append(rec)
                done.add(key)
                added += 1
        if batch:
            _append_jsonl(samples_path, batch)
    return added


def _count_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def count_run_samples(out_root: Path) -> int:
    """Count completed sample rows across all benchmarks under out_root."""
    total = 0
    if not out_root.is_dir():
        return 0
    for bench_dir in sorted(out_root.iterdir()):
        if not bench_dir.is_dir():
            continue
        keys = _load_done_keys(bench_dir / "samples.jsonl")
        for shard in bench_dir.glob("_gpu*_samples.jsonl"):
            keys |= _load_done_keys(shard)
        total += len(keys)
    return total


def write_progress(
    out_root: Path,
    *,
    benchmark: str,
    n_done: int,
    n_total: int,
    started_at: float,
    status: str = "running",
) -> None:
    """Atomic ``<out_root>/progress.json`` for owner status checks."""
    out_root.mkdir(parents=True, exist_ok=True)
    elapsed = max(0.0, time.time() - float(started_at))
    rate = (n_done / elapsed) if elapsed > 1e-3 and n_done > 0 else 0.0
    remaining = max(0, int(n_total) - int(n_done))
    eta_s = (remaining / rate) if rate > 0 else None
    payload = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": status,
        "benchmark": benchmark,
        "n_done": int(n_done),
        "n_total": int(n_total),
        "rate_per_s": round(rate, 4),
        "eta_s": None if eta_s is None else int(eta_s),
        "elapsed_s": int(elapsed),
    }
    path = out_root / "progress.json"
    tmp = out_root / "progress.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def run_job(
    job: MiniJob,
    *,
    out_root: Path,
    model_dir: Path,
    base_seed: int,
    generate_fn: Optional[GenerateFn] = None,
    batch_generate_fn: Optional[BatchGenerateFn] = None,
    dry_plan: bool = False,
    use_vllm: bool = False,
    n_gpus: int = DEFAULT_N_GPUS,
    max_num_seqs: int = DEFAULT_MAX_NUM_SEQS,
    max_new_tokens: int = STUDENT_MAX_NEW_TOKENS,
    max_model_len: int = MAX_MODEL_LEN,
    temperature: float = EVAL_TEMPERATURE,
    top_p: float = EVAL_TOP_P,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
    gpu_ids: Optional[str] = None,
    run_started_at: Optional[float] = None,
    run_n_total: Optional[int] = None,
) -> Dict[str, Any]:
    refuse_binning_seed(base_seed, context="mini_protocol")
    bench_dir = out_root / job.benchmark_id
    samples_path = bench_dir / "samples.jsonl"
    adaptor = AdaptorFactory.create_adaptor(
        job.adaptor_key, str(job.data_path), thinking_mode=False
    )
    # Absorb any prior GPU shards before planning (kill-safe resume).
    merge_shard_samples(bench_dir, samples_path)
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
        "temperature": temperature,
        "top_p": top_p,
        "base_seed": base_seed,
        "max_new_tokens": max_new_tokens,
    }
    if dry_plan:
        return plan_info

    started = float(run_started_at if run_started_at is not None else time.time())
    # Job-level total = already done + remaining; run-level total if provided.
    job_total = len(done) + len(work)
    progress_total = int(run_n_total) if run_n_total is not None else job_total

    if use_vllm:
        if work:
            write_progress(
                out_root,
                benchmark=job.benchmark_id,
                n_done=count_run_samples(out_root),
                n_total=progress_total,
                started_at=started,
            )
            _run_vllm_dp(
                work,
                bench_dir=bench_dir,
                benchmark_id=job.benchmark_id,
                adaptor_key=job.adaptor_key,
                data_path=str(job.data_path),
                model_dir=model_dir,
                n_gpus=n_gpus,
                max_num_seqs=max_num_seqs,
                max_new_tokens=max_new_tokens,
                max_model_len=max_model_len,
                temperature=temperature,
                top_p=top_p,
                gpu_memory_utilization=gpu_memory_utilization,
                gpu_ids=gpu_ids,
                out_root=out_root,
                progress_total=progress_total,
                started_at=started,
            )
            merge_shard_samples(bench_dir, samples_path)
    elif batch_generate_fn is not None:
        if work:
            texts = batch_generate_fn(work)
            _score_and_append(
                adaptor,
                work,
                texts,
                samples_path=samples_path,
                benchmark_id=job.benchmark_id,
            )
    elif generate_fn is not None:
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
    else:
        return plan_info

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
    plan_info["n_todo"] = 0
    plan_info["n_done"] = len(_load_done_keys(samples_path))
    write_progress(
        out_root,
        benchmark=job.benchmark_id,
        n_done=count_run_samples(out_root),
        n_total=progress_total,
        started_at=started,
        status="running",
    )
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


def _resolve_gpu_ids(n_gpus: int, gpu_ids: Optional[str] = None) -> List[int]:
    if gpu_ids:
        ids = [int(x) for x in gpu_ids.split(",") if x.strip() != ""]
        if not ids:
            raise ValueError("empty --gpu-ids")
        return ids
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        ids = [int(x) for x in visible.split(",") if x.strip() != ""]
        if len(ids) >= n_gpus:
            return ids[:n_gpus]
        if ids:
            return ids
    return list(range(n_gpus))


def _run_vllm_dp(
    work: Sequence[Mapping[str, Any]],
    *,
    bench_dir: Path,
    benchmark_id: str,
    adaptor_key: str,
    data_path: str,
    model_dir: Path,
    n_gpus: int,
    max_num_seqs: int,
    max_new_tokens: int,
    max_model_len: int,
    temperature: float,
    top_p: float,
    gpu_memory_utilization: float,
    gpu_ids: Optional[str] = None,
    out_root: Optional[Path] = None,
    progress_total: int = 0,
    started_at: Optional[float] = None,
) -> None:
    """Shard work across GPUs; one vLLM engine per GPU (TP=1), like rollout_passk."""
    bench_dir.mkdir(parents=True, exist_ok=True)
    ids = _resolve_gpu_ids(n_gpus, gpu_ids)
    n_gpus = len(ids)
    buckets: List[List[Dict[str, Any]]] = [[] for _ in range(n_gpus)]
    for i, item in enumerate(work):
        row = {
            "problem_id": item["problem_id"],
            "sample_idx": item["sample_idx"],
            "seed": item["seed"],
            "prompt": item["prompt"],
            "ground_truth": item["ground_truth"],
        }
        buckets[i % n_gpus].append(row)

    procs: List[subprocess.Popen] = []
    python = sys.executable
    progress_root = out_root or bench_dir.parent
    t0 = float(started_at if started_at is not None else time.time())
    for rank, bucket in enumerate(buckets):
        if not bucket:
            continue
        list_path = bench_dir / f"_gpu{rank}_work.json"
        list_path.write_text(json.dumps(bucket), encoding="utf-8")
        out_shard = bench_dir / f"_gpu{rank}_samples.jsonl"
        cmd = [
            python,
            "-m",
            "opd_eval.mini_protocol",
            "--worker",
            "--work-list",
            str(list_path),
            "--shard-out",
            str(out_shard),
            "--model-dir",
            str(model_dir),
            "--out-root",
            str(progress_root),
            "--max-num-seqs",
            str(max_num_seqs),
            "--max-new-tokens",
            str(max_new_tokens),
            "--max-model-len",
            str(max_model_len),
            "--temperature",
            str(temperature),
            "--top-p",
            str(top_p),
            "--gpu-memory-utilization",
            str(gpu_memory_utilization),
            "--benchmark-id",
            benchmark_id,
            "--worker-adaptor-key",
            adaptor_key,
            "--worker-data-path",
            data_path,
            "--progress-total",
            str(int(progress_total)),
            "--progress-started-at",
            str(t0),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(ids[rank])
        env["PYTHONUNBUFFERED"] = "1"
        print(
            f"mini_protocol gpu {rank} (physical {ids[rank]}): "
            f"{len(bucket)} requests → {out_shard}",
            flush=True,
        )
        procs.append(subprocess.Popen(cmd, env=env))

    stop = threading.Event()

    def _poll_progress() -> None:
        while not stop.wait(60.0):
            write_progress(
                progress_root,
                benchmark=benchmark_id,
                n_done=count_run_samples(progress_root),
                n_total=int(progress_total) if progress_total else count_run_samples(progress_root),
                started_at=t0,
            )

    poller = threading.Thread(target=_poll_progress, daemon=True)
    poller.start()
    rc = 0
    for proc in procs:
        rc = max(rc, int(proc.wait()))
    stop.set()
    write_progress(
        progress_root,
        benchmark=benchmark_id,
        n_done=count_run_samples(progress_root),
        n_total=int(progress_total) if progress_total else count_run_samples(progress_root),
        started_at=t0,
    )
    if rc != 0:
        raise RuntimeError(f"mini_protocol vLLM worker(s) failed rc={rc}")


def _vllm_generate_batch(
    prompts: Sequence[str],
    seeds: Sequence[int],
    *,
    model_dir: Path,
    max_num_seqs: int,
    max_new_tokens: int,
    max_model_len: int,
    temperature: float,
    top_p: float,
    gpu_memory_utilization: float,
) -> List[str]:
    """One ``llm.generate`` call with per-request SamplingParams.seed."""
    from vllm import LLM, SamplingParams

    if len(prompts) != len(seeds):
        raise ValueError("prompts/seeds length mismatch")
    if not prompts:
        return []

    llm = LLM(
        model=str(model_dir),
        tensor_parallel_size=1,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=int(max_model_len),
        gpu_memory_utilization=float(gpu_memory_utilization),
        max_num_seqs=int(max_num_seqs),
    )
    stops = vllm_stop_token_ids()
    params_list = [
        SamplingParams(
            temperature=float(temperature),
            top_p=float(top_p),
            max_tokens=int(max_new_tokens),
            n=1,
            seed=int(seed),
            stop_token_ids=stops,
            include_stop_str_in_output=True,
        )
        for seed in seeds
    ]
    outputs = llm.generate(list(prompts), params_list)
    if len(outputs) != len(prompts):
        raise RuntimeError(
            f"vLLM returned {len(outputs)} outputs for {len(prompts)} requests"
        )
    return [out.outputs[0].text for out in outputs]


def worker_main(args: argparse.Namespace) -> int:
    """Single-GPU worker: one engine, batched generate chunks, append shard jsonl."""
    work = json.loads(Path(args.work_list).read_text(encoding="utf-8"))
    if not work:
        return 0
    shard_out = Path(args.shard_out)
    done = _load_done_keys(shard_out)
    pending = [
        w
        for w in work
        if (str(w["problem_id"]), int(w["sample_idx"])) not in done
    ]
    if not pending:
        print(f"worker shard complete: {shard_out}", flush=True)
        return 0

    adaptor_key = getattr(args, "worker_adaptor_key", "") or ""
    data_path = getattr(args, "worker_data_path", "") or ""
    adaptor: Optional[BaseAdaptor] = None
    if adaptor_key and data_path:
        adaptor = AdaptorFactory.create_adaptor(
            adaptor_key, data_path, thinking_mode=False
        )

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(args.model_dir),
        tensor_parallel_size=1,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=int(args.max_model_len),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        max_num_seqs=int(args.max_num_seqs),
    )
    stops = vllm_stop_token_ids()
    # Chunk so a kill mid-job loses at most one flush window, not the whole shard.
    chunk = max(int(args.max_num_seqs), 64)
    n_written = 0
    for start in range(0, len(pending), chunk):
        batch_work = pending[start : start + chunk]
        params_list = [
            SamplingParams(
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                max_tokens=int(args.max_new_tokens),
                n=1,
                seed=int(w["seed"]),
                stop_token_ids=stops,
                include_stop_str_in_output=True,
            )
            for w in batch_work
        ]
        outputs = llm.generate([w["prompt"] for w in batch_work], params_list)
        if len(outputs) != len(batch_work):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} outputs for {len(batch_work)} requests"
            )
        rows: List[Dict[str, Any]] = []
        for w, out in zip(batch_work, outputs):
            text = out.outputs[0].text
            if adaptor is not None:
                scored = score_response(adaptor, text, w["ground_truth"])
            else:
                scored = {
                    "response": text,
                    "extracted": "",
                    "verified": False,
                    "n_tokens": len(text.split()),
                }
            rows.append(
                {
                    "problem_id": w["problem_id"],
                    "sample_idx": w["sample_idx"],
                    "seed": w["seed"],
                    "benchmark_id": args.benchmark_id,
                    "g": 0,
                    **scored,
                }
            )
        _append_jsonl(shard_out, rows)
        n_written += len(rows)
        print(
            f"worker flush +{len(rows)} (total {n_written}/{len(pending)}) → {shard_out}",
            flush=True,
        )
        # Refresh owner-visible progress every flush (≈ every max_num_seqs gens).
        progress_total = int(getattr(args, "progress_total", 0) or 0)
        started_at = float(getattr(args, "progress_started_at", 0) or 0)
        if progress_total > 0 and started_at > 0:
            write_progress(
                Path(args.out_root),
                benchmark=str(args.benchmark_id),
                n_done=count_run_samples(Path(args.out_root)),
                n_total=progress_total,
                started_at=started_at,
            )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--base-seed", type=int, default=EVAL_SEED_MINI_PROTOCOL)
    p.add_argument(
        "--h-hard-mini",
        type=Path,
        default=DEFAULT_H_HARD_MINI_PATH,
        help=f"default: {DEFAULT_H_HARD_MINI_PATH}",
    )
    p.add_argument(
        "--h-mini",
        type=Path,
        default=DEFAULT_H_MINI_PATH,
        help=f"default: {DEFAULT_H_MINI_PATH}",
    )
    p.add_argument(
        "--math500-mini",
        type=Path,
        default=DEFAULT_MATH500_MINI_PATH,
        help=f"default: {DEFAULT_MATH500_MINI_PATH}",
    )
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
    p.add_argument("--n-gpus", type=int, default=DEFAULT_N_GPUS)
    p.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="Comma-separated physical GPU ids (overrides --n-gpus count)",
    )
    p.add_argument("--max-num-seqs", type=int, default=DEFAULT_MAX_NUM_SEQS)
    p.add_argument("--max-new-tokens", type=int, default=STUDENT_MAX_NEW_TOKENS)
    p.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    p.add_argument("--temperature", type=float, default=EVAL_TEMPERATURE)
    p.add_argument("--top-p", type=float, default=EVAL_TOP_P)
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
    )
    # Worker / internal flags
    p.add_argument("--worker", action="store_true")
    p.add_argument("--work-list", type=Path, default=None)
    p.add_argument("--shard-out", type=Path, default=None)
    p.add_argument("--benchmark-id", type=str, default="")
    p.add_argument("--adaptor-key", type=str, default="")
    p.add_argument("--worker-adaptor-key", type=str, default="")
    p.add_argument("--worker-data-path", type=str, default="")
    p.add_argument("--progress-total", type=int, default=0)
    p.add_argument("--progress-started-at", type=float, default=0.0)
    return p


def _plan_run_total(jobs: Sequence[MiniJob], base_seed: int) -> int:
    """Total sample count for ETA (resume-aware: includes already-done)."""
    total = 0
    for job in jobs:
        if not Path(job.data_path).is_file():
            continue
        adaptor = AdaptorFactory.create_adaptor(
            job.adaptor_key, str(job.data_path), thinking_mode=False
        )
        n_problems = 0
        for item in adaptor.data:
            if job.bin_exclude is not None and str(item.get("bin", "")) == job.bin_exclude:
                continue
            n_problems += 1
        total += n_problems * int(job.n)
    return total


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.worker:
        if args.work_list is None or args.shard_out is None:
            raise SystemExit("--worker requires --work-list and --shard-out")
        return worker_main(args)

    refuse_binning_seed(int(args.base_seed), context="mini_protocol")
    jobs = default_mini_jobs(
        h_hard_mini=args.h_hard_mini,
        h_mini=args.h_mini,
        math500_mini=args.math500_mini,
        aime_union=args.aime_union,
        amc23=args.amc23,
        hmmt25=args.hmmt25,
    )
    if args.dry_plan:
        reports = []
        for job in jobs:
            reports.append(
                run_job(
                    job,
                    out_root=args.out_root,
                    model_dir=args.model_dir,
                    base_seed=int(args.base_seed),
                    dry_plan=True,
                )
            )
        print(json.dumps({"jobs": reports}, indent=2))
        return 0

    run_total = _plan_run_total(jobs, int(args.base_seed))
    # Resume: subtract nothing from ETA total — n_done starts from disk count.
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
    print(json.dumps({"jobs": reports}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
