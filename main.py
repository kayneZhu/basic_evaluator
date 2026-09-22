import json
import os
os.environ['OMP_NUM_THREADS'] = '1'


class Config:
    MODEL_PATH = "../models/Qwen3-1.7B"
    #MODEL_PATH = "../models/math_baseline_3k_global_step_1000"
    # MODEL_PATH = "../models/C3_math_mixture_hint_global_step_1000"
    # MODEL_PATH = "../models/ttn2k_unsolvable_hint_dapo_8xpro6000_global_step_1200"

    OUTPUT_DIR = None  # auto-generated

    BENCHMARK_DATA_PATH = "./data/ttn_test_200.jsonl"
    BENCHMARK_TYPE = "math500_bench_schema"

    THINKING_MODE = True

    TENSOR_PARALLEL_SIZE = 1
    DATA_PARALLEL_SIZE = 4  # 几 个独立进程，各在 4 张 GPU 上推理
    GPU_MEMORY_UTILIZATION = 0.95
    MAX_MODEL_LEN = 10000  # 输入+输出总长度限制

    USE_PARALLEL = True
    MAX_NUM_SEQS = 256
    # 大 batch 减少 generate() 调用次数开销；vLLM 内部靠 MAX_NUM_SEQS 控制并发
    BATCH_SIZE = 256

    MAX_TOKENS = 8192  # 输出长度限制
    TEMPERATURE = 1
    TOP_P = 1
    STOP_TOKENS = None  # string stops only; token-id stops come from the contract

    MAX_SAMPLE = None
    PASS_N = 1
    REPEAT_N = 1  # 重复评测次数（TEMPERATURE>0 时生效，支持 Pass@K 多次均值）

    # Probe100 guidance config (set per-job via run_batch.py or here)
    G_LEVELS = None
    GUIDANCE_MODES = None


def _print_config():
    print("=" * 60)
    print("LLM Evaluator - Starting Evaluation")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Model Path:      {Config.MODEL_PATH}")
    print(f"  Benchmark:       {Config.BENCHMARK_TYPE}  ({Config.BENCHMARK_DATA_PATH})")
    print(f"  Thinking Mode:   {Config.THINKING_MODE}")
    print(f"  Pass N:          {Config.PASS_N}")
    print(f"  TP: {Config.TENSOR_PARALLEL_SIZE}  DP: {Config.DATA_PARALLEL_SIZE}")
    print(f"  Batch Size:      {Config.BATCH_SIZE}")
    print(f"  Temperature:     {Config.TEMPERATURE}")
    print(f"  Top P:           {Config.TOP_P}")
    print(f"  Max Tokens:      {Config.MAX_TOKENS}")
    from opd_eval.contract import vllm_stop_token_ids
    print(f"  Stop token ids:  {vllm_stop_token_ids()}  (contract; not tokenizer eos)")
    print(f"  Max Samples:     {Config.MAX_SAMPLE if Config.MAX_SAMPLE is not None else 'All'}")
    print(f"  Output Dir:      {Config.OUTPUT_DIR}")
    if _repeat_enabled():
        print(f"  Repeat N:        {Config.REPEAT_N}")
    if Config.BENCHMARK_TYPE == 'probe100':
        print(f"  G Levels:        {Config.G_LEVELS}")
        print(f"  Guidance Modes:  {Config.GUIDANCE_MODES}")


def _get_gpu_ids(dp_size: int):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        ids = [int(x.strip()) for x in visible.split(",") if x.strip()]
    else:
        ids = list(range(dp_size))
    if len(ids) < dp_size:
        raise RuntimeError(
            f"DATA_PARALLEL_SIZE={dp_size} but only {len(ids)} GPUs visible "
            f"(CUDA_VISIBLE_DEVICES={visible or 'not set'})")
    return ids[:dp_size]


def _adaptor_kwargs():
    """Extra kwargs forwarded to the adaptor constructor (benchmark-specific)."""
    if Config.BENCHMARK_TYPE == 'probe100':
        return {
            'g_levels': Config.G_LEVELS,
            'guidance_modes': Config.GUIDANCE_MODES,
            'max_raw_samples': Config.MAX_SAMPLE,
        }
    return {}


def _worker_config():
    return {
        'model_path': Config.MODEL_PATH,
        'benchmark_type': Config.BENCHMARK_TYPE,
        'data_path': Config.BENCHMARK_DATA_PATH,
        'thinking_mode': Config.THINKING_MODE,
        'tensor_parallel_size': Config.TENSOR_PARALLEL_SIZE,
        'gpu_memory_utilization': Config.GPU_MEMORY_UTILIZATION,
        'max_model_len': Config.MAX_MODEL_LEN,
        'enable_thinking': Config.THINKING_MODE,
        'batch_size': Config.BATCH_SIZE,
        'max_tokens': Config.MAX_TOKENS,
        'temperature': Config.TEMPERATURE,
        'top_p': Config.TOP_P,
        'stop': Config.STOP_TOKENS,
        'n_samples': Config.PASS_N,
        # For probe100, MAX_SAMPLE is handled inside the adaptor (pre-expansion);
        # pass None so evaluator/dp_worker don't truncate the expanded data.
        'max_samples': None if Config.BENCHMARK_TYPE == 'probe100' else Config.MAX_SAMPLE,
        'output_dir': Config.OUTPUT_DIR,
        'max_num_seqs': Config.MAX_NUM_SEQS,
        'adaptor_kwargs': _adaptor_kwargs(),
    }


# ── Progress monitor ─────────────────────────────────────────────────────────

def _monitor_progress(procs, output_dir, dp_size, config):
    """Poll shard files and show unified progress until all workers finish."""
    import time as _time

    max_samples = config.get('max_samples')

    # Compute total expected samples (same logic as dp_worker shard split)
    from adaptors.adaptor_factory import AdaptorFactory
    akw = config.get('adaptor_kwargs', {})
    adaptor = AdaptorFactory.create_adaptor(
        config['benchmark_type'], config['data_path'], config['thinking_mode'], **akw)
    total = len(adaptor.load_benchmark_data())
    if max_samples and max_samples > 0:
        total = min(total, max_samples)
    del adaptor

    start = _time.time()
    last_done = 0

    while True:
        # Count completed items across all shards
        done = 0
        for rank in range(dp_size):
            path = os.path.join(output_dir, f"_shard{rank}_results.jsonl")
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        done += sum(1 for _ in f)
                except OSError:
                    pass

        elapsed = _time.time() - start
        pct = done / total * 100 if total > 0 else 0
        speed = done / elapsed if elapsed > 0 else 0
        remaining = total - done
        eta_min = (remaining / speed / 60) if speed > 0 else 0

        bar_len = 30
        filled = int(bar_len * done / total) if total > 0 else 0
        bar = '█' * filled + '░' * (bar_len - filled)

        print(f"\r  [{bar}] {done}/{total} ({pct:.1f}%) | "
              f"{speed:.1f} samples/s | ETA: {eta_min:.1f} min   ",
              end='', flush=True)

        last_done = done

        # Check if all workers have exited
        if all(not p.is_alive() for p in procs):
            break

        _time.sleep(5)

    # Final line
    done = 0
    for rank in range(dp_size):
        path = os.path.join(output_dir, f"_shard{rank}_results.jsonl")
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    done += sum(1 for _ in f)
            except OSError:
                pass
    elapsed = _time.time() - start
    pct = done / total * 100 if total > 0 else 0
    print(f"\r  [{'█' * bar_len}] {done}/{total} ({pct:.1f}%) | "
          f"Total: {elapsed / 60:.1f} min                        ")

    for p in procs:
        p.join()


# ── Data-parallel path (DP > 1) ─────────────────────────────────────────────

def run_data_parallel():
    from multiprocessing import Process
    from core.dp_worker import dp_worker_main, load_jsonl
    from datetime import datetime

    dp_size = Config.DATA_PARALLEL_SIZE
    gpu_ids = _get_gpu_ids(dp_size)
    config = _worker_config()

    print(f"\n  Launching {dp_size} workers on GPUs {gpu_ids} ...")
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    procs = []
    for rank in range(dp_size):
        p = Process(target=dp_worker_main,
                    args=(rank, gpu_ids[rank], dp_size, config))
        p.start()
        procs.append(p)

    # Unified progress monitor: poll shard files every 5s
    _monitor_progress(procs, Config.OUTPUT_DIR, dp_size, config)

    failed = [i for i, p in enumerate(procs) if p.exitcode != 0]
    if failed:
        print(f"\nERROR: Workers {failed} exited with errors. "
              "Fix the issue and re-run — completed shards will be resumed.")
        return None

    # ── Merge shard files ────────────────────────────────────────────────
    print("\nMerging shard results ...")
    all_results = []
    all_pass_rates = []
    for rank in range(dp_size):
        all_results.extend(load_jsonl(
            os.path.join(Config.OUTPUT_DIR, f"_shard{rank}_results.jsonl")))
        all_pass_rates.extend(load_jsonl(
            os.path.join(Config.OUTPUT_DIR, f"_shard{rank}_pass_rates.jsonl")))

    results_path = os.path.join(Config.OUTPUT_DIR, "results.jsonl")
    pass_path = os.path.join(Config.OUTPUT_DIR,
                             f"per_question_pass_PASS{Config.PASS_N}.jsonl")

    with open(results_path, 'w', encoding='utf-8') as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    with open(pass_path, 'w', encoding='utf-8') as f:
        for r in all_pass_rates:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # ── Summary report ───────────────────────────────────────────────────
    total = len(all_results)
    correct = sum(1 for r in all_results if r.get('is_correct'))
    accuracy = correct / total * 100 if total > 0 else 0

    report = {
        'timestamp': datetime.now().isoformat(),
        'total_samples': total,
        'correct_samples': correct,
        'incorrect_samples': total - correct,
        'accuracy': correct / total if total > 0 else 0,
        'accuracy_percentage': accuracy,
        'data_parallel_size': dp_size,
        'pass_n': Config.PASS_N,
    }
    ts = report['timestamp'].replace(':', '-').replace('.', '-')
    report_path = os.path.join(Config.OUTPUT_DIR, f"report_{ts}.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"  Total Samples:   {total}")
    print(f"  Correct:         {correct}")
    print(f"  Accuracy:        {accuracy:.2f}%")
    print(f"\n  Results:    {results_path}")
    print(f"  Pass rates: {pass_path}")
    print(f"  Report:     {report_path}")
    print("=" * 60)

    return accuracy


# ── Single-process path (DP = 1) ────────────────────────────────────────────

def run_single_process():
    from core.evaluator import Evaluator
    from adaptors.adaptor_factory import AdaptorFactory

    print("\nInitializing adaptor ...")
    adaptor = AdaptorFactory.create_adaptor(
        Config.BENCHMARK_TYPE, Config.BENCHMARK_DATA_PATH, Config.THINKING_MODE,
        **_adaptor_kwargs())

    print("Initializing evaluator ...")
    evaluator = Evaluator(
        model_path=Config.MODEL_PATH,
        tensor_parallel_size=Config.TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=Config.GPU_MEMORY_UTILIZATION,
        max_model_len=Config.MAX_MODEL_LEN,
        use_parallel=Config.USE_PARALLEL,
        batch_size=Config.BATCH_SIZE,
        enable_thinking=Config.THINKING_MODE,
        max_num_seqs=Config.MAX_NUM_SEQS
    )

    print("\nRunning evaluation ...")
    effective_max = None if Config.BENCHMARK_TYPE == 'probe100' else Config.MAX_SAMPLE
    evaluation_result = evaluator.evaluate(
        adaptor=adaptor, max_tokens=Config.MAX_TOKENS,
        temperature=Config.TEMPERATURE, top_p=Config.TOP_P,
        stop=Config.STOP_TOKENS, max_samples=effective_max,
        n_samples=Config.PASS_N, output_dir=Config.OUTPUT_DIR
    )

    report = evaluation_result['report']

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"  Total Samples:   {report['total_samples']}")
    print(f"  Correct:         {report['correct_samples']}")
    print(f"  Accuracy:        {report['accuracy_percentage']:.2f}%")

    ts = report['timestamp'].replace(':', '-').replace('.', '-')
    report_path = os.path.join(Config.OUTPUT_DIR, f"report_{ts}.json")
    evaluator.save_report(report, report_path)

    print(f"\n  Results:    {os.path.join(Config.OUTPUT_DIR, 'results.jsonl')}")
    print(f"  Pass rates: {os.path.join(Config.OUTPUT_DIR, f'per_question_pass_PASS{Config.PASS_N}.jsonl')}")
    print(f"  Report:     {report_path}")
    print("=" * 60)

    evaluator.cleanup()
    return report['accuracy_percentage']


# ── Repeat-evaluation helpers ────────────────────────────────────────────────

def _repeat_enabled():
    return Config.REPEAT_N > 1 and Config.TEMPERATURE > 0


def _run_once():
    """Dispatch a single evaluation run; returns accuracy (%) or None on error."""
    if Config.DATA_PARALLEL_SIZE > 1:
        return run_data_parallel()
    return run_single_process()


def _run_single_process_repeat():
    """Repeat evaluation N times with vLLM loaded only once (DP=1)."""
    from core.evaluator import Evaluator
    from adaptors.adaptor_factory import AdaptorFactory

    repeat_n = Config.REPEAT_N
    base_output_dir = Config.OUTPUT_DIR

    print("\nInitializing adaptor ...")
    adaptor = AdaptorFactory.create_adaptor(
        Config.BENCHMARK_TYPE, Config.BENCHMARK_DATA_PATH, Config.THINKING_MODE,
        **_adaptor_kwargs())

    print("Initializing evaluator (model loaded once for all repeats) ...")
    evaluator = Evaluator(
        model_path=Config.MODEL_PATH,
        tensor_parallel_size=Config.TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=Config.GPU_MEMORY_UTILIZATION,
        max_model_len=Config.MAX_MODEL_LEN,
        use_parallel=Config.USE_PARALLEL,
        batch_size=Config.BATCH_SIZE,
        enable_thinking=Config.THINKING_MODE,
        max_num_seqs=Config.MAX_NUM_SEQS
    )

    effective_max = None if Config.BENCHMARK_TYPE == 'probe100' else Config.MAX_SAMPLE
    accuracies = []

    for run_idx in range(1, repeat_n + 1):
        run_dir = os.path.join(base_output_dir, f"run_{run_idx}")
        Config.OUTPUT_DIR = run_dir

        print(f"\n{'─' * 60}")
        print(f"  Run {run_idx}/{repeat_n}  →  {run_dir}")
        print(f"{'─' * 60}")

        evaluation_result = evaluator.evaluate(
            adaptor=adaptor, max_tokens=Config.MAX_TOKENS,
            temperature=Config.TEMPERATURE, top_p=Config.TOP_P,
            stop=Config.STOP_TOKENS, max_samples=effective_max,
            n_samples=Config.PASS_N, output_dir=run_dir
        )

        report = evaluation_result['report']
        acc = report['accuracy_percentage']
        accuracies.append(acc)

        ts = report['timestamp'].replace(':', '-').replace('.', '-')
        report_path = os.path.join(run_dir, f"report_{ts}.json")
        evaluator.save_report(report, report_path)

        print(f"  Run {run_idx}: {report['total_samples']} samples, "
              f"{report['correct_samples']} correct, accuracy {acc:.2f}%")

    evaluator.cleanup()
    Config.OUTPUT_DIR = base_output_dir
    return accuracies


def _run_data_parallel_repeat():
    """Repeat evaluation N times with each DP worker loading model only once."""
    from multiprocessing import Process
    from core.dp_worker import dp_worker_main, load_jsonl
    from datetime import datetime

    repeat_n = Config.REPEAT_N
    base_output_dir = Config.OUTPUT_DIR
    dp_size = Config.DATA_PARALLEL_SIZE
    gpu_ids = _get_gpu_ids(dp_size)

    config = _worker_config()
    config['repeat_n'] = repeat_n
    config['output_dir'] = base_output_dir

    print(f"\n  Launching {dp_size} workers on GPUs {gpu_ids} ...")
    print(f"  Each worker will run {repeat_n} evaluations (model loaded once).")
    os.makedirs(base_output_dir, exist_ok=True)

    procs = []
    for rank in range(dp_size):
        p = Process(target=dp_worker_main,
                    args=(rank, gpu_ids[rank], dp_size, config))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    failed = [i for i, p in enumerate(procs) if p.exitcode != 0]
    if failed:
        print(f"\nERROR: Workers {failed} exited with errors.")

    accuracies = []
    for run_idx in range(1, repeat_n + 1):
        run_dir = os.path.join(base_output_dir, f"run_{run_idx}")
        all_results = []
        all_pass_rates = []
        for rank in range(dp_size):
            all_results.extend(load_jsonl(
                os.path.join(run_dir, f"_shard{rank}_results.jsonl")))
            all_pass_rates.extend(load_jsonl(
                os.path.join(run_dir, f"_shard{rank}_pass_rates.jsonl")))

        results_path = os.path.join(run_dir, "results.jsonl")
        pass_path = os.path.join(run_dir,
                                 f"per_question_pass_PASS{Config.PASS_N}.jsonl")
        with open(results_path, 'w', encoding='utf-8') as f:
            for r in all_results:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        with open(pass_path, 'w', encoding='utf-8') as f:
            for r in all_pass_rates:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

        total = len(all_results)
        correct = sum(1 for r in all_results if r.get('is_correct'))
        accuracy = correct / total * 100 if total > 0 else 0
        accuracies.append(accuracy)

        report = {
            'timestamp': datetime.now().isoformat(),
            'total_samples': total,
            'correct_samples': correct,
            'incorrect_samples': total - correct,
            'accuracy': correct / total if total > 0 else 0,
            'accuracy_percentage': accuracy,
            'data_parallel_size': dp_size,
            'pass_n': Config.PASS_N,
            'repeat_run': run_idx,
        }
        ts = report['timestamp'].replace(':', '-').replace('.', '-')
        report_path = os.path.join(run_dir, f"report_{ts}.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"  Run {run_idx}: {total} samples, {correct} correct, "
              f"accuracy {accuracy:.2f}%")

    Config.OUTPUT_DIR = base_output_dir
    return accuracies


def _print_repeat_summary(accuracies, base_output_dir):
    """Print and save the repeat-evaluation summary."""
    import math
    from datetime import datetime

    pass_n = Config.PASS_N
    label = f"Pass@{pass_n}" if pass_n > 1 else "Accuracy"

    print(f"\n{'=' * 60}")
    print(f"Repeat Evaluation Summary  ({label}, {len(accuracies)} runs)")
    print(f"{'=' * 60}")

    if not accuracies:
        print("  No successful runs.")
        return

    mean_acc = sum(accuracies) / len(accuracies)
    if len(accuracies) > 1:
        variance = sum((a - mean_acc) ** 2 for a in accuracies) / (len(accuracies) - 1)
        std_acc = math.sqrt(variance)
    else:
        std_acc = 0.0

    for i, acc in enumerate(accuracies, 1):
        print(f"  Run {i}: {label} = {acc:.2f}%")
    print(f"{'─' * 40}")
    print(f"  Mean {label}:  {mean_acc:.2f}%")
    print(f"  Std Deviation:   {std_acc:.2f}%")
    print(f"  Min / Max:       {min(accuracies):.2f}% / {max(accuracies):.2f}%")
    print(f"  Successful Runs: {len(accuracies)}/{Config.REPEAT_N}")
    print(f"{'=' * 60}")

    summary = {
        'timestamp': datetime.now().isoformat(),
        'repeat_n': Config.REPEAT_N,
        'pass_n': pass_n,
        'metric': label,
        'successful_runs': len(accuracies),
        'accuracies': accuracies,
        'mean_accuracy_percentage': mean_acc,
        'std_accuracy_percentage': std_acc,
        'min_accuracy_percentage': min(accuracies),
        'max_accuracy_percentage': max(accuracies),
    }
    os.makedirs(base_output_dir, exist_ok=True)
    summary_path = os.path.join(base_output_dir, "repeat_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Summary saved:   {summary_path}")


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    model_name = os.path.basename(Config.MODEL_PATH.rstrip('/'))
    bench_name = os.path.splitext(os.path.basename(Config.BENCHMARK_DATA_PATH))[0]
    base_output_dir = f"./outputs/{bench_name}/{model_name}_PASS{Config.PASS_N}_{Config.MAX_TOKENS}"

    if Config.BENCHMARK_TYPE == 'probe100' and Config.G_LEVELS:
        g_tag = '_'.join(f"g{g}" for g in Config.G_LEVELS)
        base_output_dir += f"_{g_tag}"

    Config.OUTPUT_DIR = base_output_dir

    _print_config()

    if not os.path.exists(Config.MODEL_PATH):
        print(f"\nError: Model path does not exist: {Config.MODEL_PATH}")
        return
    if not os.path.exists(Config.BENCHMARK_DATA_PATH):
        print(f"\nError: Benchmark data path does not exist: {Config.BENCHMARK_DATA_PATH}")
        return

    if not _repeat_enabled():
        _run_once()
        return

    # ── Repeat-N: model loaded once, evaluate() called N times ───────────
    print(f"\n{'=' * 60}")
    print(f"Repeat Evaluation: {Config.REPEAT_N} runs "
          f"(Pass@{Config.PASS_N}, TEMPERATURE={Config.TEMPERATURE})")
    print(f"{'=' * 60}")

    if Config.DATA_PARALLEL_SIZE > 1:
        accuracies = _run_data_parallel_repeat()
    else:
        accuracies = _run_single_process_repeat()

    _print_repeat_summary(accuracies, base_output_dir)


if __name__ == "__main__":
    main()
