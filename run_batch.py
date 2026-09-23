#!/usr/bin/env python3
"""
Batch evaluation runner — run multiple (model, benchmark) combinations
in sequence without stopping to edit main.py.

Usage:
    python run_batch.py

Edit the JOBS list below to define your evaluation matrix.
Each job can override any Config field; unspecified fields keep
the defaults from main.py's Config class.
"""
import os
import sys
import time
from datetime import datetime

# ── Define your evaluation jobs here ─────────────────────────────────────────

# Probe100 guidance experiment:
#   ttn_unsolvable_pass64 × Qwen3-1.7B
#   5 g_levels × 2 modes (prefix+hint) × Pass@8 × 4 repeats
_PROBE100_BASE = {
    "MODEL_PATH": "../models/Qwen3-1.7B",
    "BENCHMARK_DATA_PATH": "./data/ttn_unsolvable_pass64.jsonl",
    "BENCHMARK_TYPE": "probe100",
    "GUIDANCE_MODES": ["hint"],
    "PASS_N": 8,
    "REPEAT_N": 4,
    "TEMPERATURE": 1,
    "TOP_P": 1,
    "TENSOR_PARALLEL_SIZE": 1,
    "DATA_PARALLEL_SIZE": 1,
    "GPU_MEMORY_UTILIZATION": 0.95,
    "MAX_NUM_SEQS": 256,
    "BATCH_SIZE": 32,
    "MAX_MODEL_LEN": 11264,
    "MAX_TOKENS": 10240,
}

JOBS = [{**_PROBE100_BASE, "G_LEVELS": [g]} for g in [0, 0.25, 0.5, 0.75, 1.0]]

# ── Runner logic ─────────────────────────────────────────────────────────────

def run_batch():
    from main import Config, main as run_main

    defaults = {k: v for k, v in vars(Config).items()
                if not k.startswith('_') and k.isupper()}

    total_jobs = len(JOBS)
    results = []

    print("╔" + "═" * 68 + "╗")
    print(f"║  Batch Runner: {total_jobs} job(s) queued" + " " * (68 - 34 - len(str(total_jobs))) + "║")
    print("╚" + "═" * 68 + "╝")

    batch_start = time.time()

    for idx, job in enumerate(JOBS, 1):
        for k, v in defaults.items():
            setattr(Config, k, v)
        for k, v in job.items():
            if not hasattr(Config, k):
                print(f"  WARNING: unknown config key {k!r}, skipping")
                continue
            setattr(Config, k, v)

        model_name = os.path.basename(Config.MODEL_PATH.rstrip('/'))
        bench_name = os.path.splitext(os.path.basename(Config.BENCHMARK_DATA_PATH))[0]
        g_info = f"  G_LEVELS: {Config.G_LEVELS}" if Config.BENCHMARK_TYPE == 'probe100' else ""

        print(f"\n{'━' * 70}")
        print(f"  Job {idx}/{total_jobs}")
        print(f"  Model:     {model_name}")
        print(f"  Benchmark: {bench_name}  (type: {Config.BENCHMARK_TYPE})")
        print(f"  Temp: {Config.TEMPERATURE}  Repeat: {Config.REPEAT_N}  PassN: {Config.PASS_N}")
        if g_info:
            print(g_info)
        print(f"{'━' * 70}")

        job_start = time.time()
        try:
            run_main()
            status = "OK"
        except Exception as e:
            print(f"\n  ERROR in job {idx}: {e}")
            status = f"FAILED: {e}"
        job_elapsed = time.time() - job_start

        results.append({
            "job": idx,
            "model": model_name,
            "benchmark": bench_name,
            "benchmark_type": Config.BENCHMARK_TYPE,
            "g_levels": getattr(Config, 'G_LEVELS', None),
            "output_dir": Config.OUTPUT_DIR,
            "status": status,
            "elapsed_min": round(job_elapsed / 60, 1),
        })

    # ── Final summary ────────────────────────────────────────────────────
    batch_elapsed = time.time() - batch_start

    print(f"\n{'╔' + '═' * 68 + '╗'}")
    print(f"║  Batch Complete: {total_jobs} job(s) in {batch_elapsed / 60:.1f} min" +
          " " * max(0, 68 - 40 - len(str(total_jobs)) - len(f"{batch_elapsed / 60:.1f}")) + "║")
    print(f"{'╚' + '═' * 68 + '╝'}")

    for r in results:
        tag = "✓" if r["status"] == "OK" else "✗"
        g_str = f"  g={r['g_levels']}" if r.get('g_levels') else ""
        print(f"  {tag} Job {r['job']}: {r['model']} × {r['benchmark']}{g_str}  "
              f"[{r['elapsed_min']} min]  {r['status']}")
        if r["status"] == "OK":
            print(f"    → {r['output_dir']}")

    log_path = f"./outputs/batch_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    import json
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump({"jobs": results, "total_elapsed_min": round(batch_elapsed / 60, 1)},
                  f, indent=2, ensure_ascii=False)
    print(f"\n  Batch log: {log_path}")


if __name__ == "__main__":
    run_batch()
