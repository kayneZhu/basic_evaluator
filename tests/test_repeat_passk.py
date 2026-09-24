#!/usr/bin/env python3
"""
Smoke test for REPEAT_N + Pass@K logic.

Verifies the full pipeline WITHOUT GPU by mocking the inference engine.
Run:  python tests/test_repeat_passk.py
"""
import json
import math
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Stub heavy dependencies that aren't available in test environments ────────
import types

_vllm_stubs = {
    'vllm': {'LLM': type('LLM', (), {}), 'SamplingParams': type('SamplingParams', (), {})},
    'vllm.engine': {},
    'vllm.engine.arg_utils': {'AsyncEngineArgs': type('AsyncEngineArgs', (), {})},
    'torch': {},
    'math_verify': {
        'parse': lambda *a, **kw: None,
        'verify': lambda *a, **kw: False,
        'LatexExtractionConfig': type('LatexExtractionConfig', (), {}),
        'ExprExtractionConfig': type('ExprExtractionConfig', (), {}),
        'StringExtractionConfig': type('StringExtractionConfig', (), {}),
    },
}
for mod_name, attrs in _vllm_stubs.items():
    # Never replace a real install — a hollow torch stub poisons later tests
    # that load transformers / train-side modules in the same pytest process.
    if mod_name in sys.modules:
        continue
    try:
        __import__(mod_name)
        continue
    except Exception:
        pass
    m = types.ModuleType(mod_name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[mod_name] = m



# ── Mock inference engine ────────────────────────────────────────────────────

class MockInference:
    """Return deterministic but varied answers for testing."""
    _call_count = 0

    def generate_single_batch_with_metrics(self, prompts, max_tokens=2048,
                                           temperature=0.0, top_p=1.0,
                                           stop=None, system_prompt=None,
                                           n=1, raw_prompts=False):
        import time
        results = []
        for p_idx, _prompt in enumerate(prompts):
            MockInference._call_count += 1
            if n > 1:
                texts = []
                for s in range(n):
                    seed = MockInference._call_count * 1000 + p_idx * 100 + s
                    if seed % 3 == 0:
                        texts.append("After reasoning... \\boxed{42}")
                    elif seed % 5 == 0:
                        texts.append("\\boxed{42}")
                    else:
                        texts.append("\\boxed{wrong}")
                results.append({
                    'generated_text': texts,
                    'prompt_tokens': 10,
                    'completion_tokens': 20 * n,
                    'total_tokens': 10 + 20 * n,
                })
            else:
                seed = MockInference._call_count * 1000 + p_idx
                if seed % 2 == 0:
                    txt = "\\boxed{42}"
                else:
                    txt = "\\boxed{wrong}"
                results.append({
                    'generated_text': txt,
                    'prompt_tokens': 10,
                    'completion_tokens': 20,
                    'total_tokens': 30,
                })
        return {
            'results': results,
            'metrics': {'batch_time': 0.01, 'batch_size': len(prompts),
                        'total_tokens': sum(r['total_tokens'] for r in results),
                        'tokens_per_second': 99999},
        }

    def cleanup(self):
        pass


def _make_test_data(tmp_dir, n_samples=10):
    """Create a tiny JSONL dataset where ground_truth is always '42'."""
    data_path = os.path.join(tmp_dir, "test_data.jsonl")
    with open(data_path, 'w') as f:
        for i in range(n_samples):
            f.write(json.dumps({
                "index": i,
                "question": f"What is 6*7? (problem {i})",
                "ground_truth": "42",
            }) + '\n')
    return data_path


def test_repeat_pass1():
    """REPEAT_N=3, PASS_N=1: should produce 3 accuracy values."""
    print("TEST: repeat=3, pass@1 ...", end=" ")

    from main import Config, _repeat_enabled, _print_repeat_summary
    from core.evaluator import Evaluator

    tmp = tempfile.mkdtemp()
    try:
        data_path = _make_test_data(tmp, n_samples=10)
        Config.BENCHMARK_DATA_PATH = data_path
        Config.BENCHMARK_TYPE = "math500_bench_schema"
        Config.THINKING_MODE = False
        Config.TEMPERATURE = 1.0
        Config.PASS_N = 1
        Config.REPEAT_N = 3
        Config.OUTPUT_DIR = os.path.join(tmp, "out")
        Config.DATA_PARALLEL_SIZE = 1
        Config.BATCH_SIZE = 100

        assert _repeat_enabled(), "_repeat_enabled should be True"

        from adaptors.adaptor_factory import AdaptorFactory
        adaptor = AdaptorFactory.create_adaptor(
            Config.BENCHMARK_TYPE, data_path, Config.THINKING_MODE)

        evaluator = Evaluator.__new__(Evaluator)
        evaluator.batch_size = Config.BATCH_SIZE
        evaluator.inference_engine = MockInference()

        accuracies = []
        for run_idx in range(1, Config.REPEAT_N + 1):
            run_dir = os.path.join(Config.OUTPUT_DIR, f"run_{run_idx}")
            result = evaluator.evaluate(
                adaptor=adaptor, max_tokens=100,
                temperature=Config.TEMPERATURE, top_p=1.0,
                n_samples=Config.PASS_N, output_dir=run_dir
            )
            accuracies.append(result['report']['accuracy_percentage'])

        assert len(accuracies) == 3, f"expected 3 runs, got {len(accuracies)}"
        for a in accuracies:
            assert 0 <= a <= 100, f"accuracy out of range: {a}"

        for run_idx in range(1, 4):
            rp = os.path.join(Config.OUTPUT_DIR, f"run_{run_idx}", "results.jsonl")
            assert os.path.exists(rp), f"missing {rp}"

        print(f"PASS  (accuracies: {[f'{a:.1f}%' for a in accuracies]})")
    finally:
        shutil.rmtree(tmp)


def test_repeat_passk():
    """REPEAT_N=3, PASS_N=4: should produce 3 Pass@4 values."""
    print("TEST: repeat=3, pass@4 ...", end=" ")

    from main import Config, _repeat_enabled
    from core.evaluator import Evaluator

    tmp = tempfile.mkdtemp()
    try:
        data_path = _make_test_data(tmp, n_samples=8)
        Config.BENCHMARK_DATA_PATH = data_path
        Config.BENCHMARK_TYPE = "math500_bench_schema"
        Config.THINKING_MODE = False
        Config.TEMPERATURE = 1.0
        Config.PASS_N = 4
        Config.REPEAT_N = 3
        Config.OUTPUT_DIR = os.path.join(tmp, "out")
        Config.DATA_PARALLEL_SIZE = 1
        Config.BATCH_SIZE = 100

        assert _repeat_enabled(), "_repeat_enabled should be True for PASS_N>1"

        from adaptors.adaptor_factory import AdaptorFactory
        adaptor = AdaptorFactory.create_adaptor(
            Config.BENCHMARK_TYPE, data_path, Config.THINKING_MODE)

        evaluator = Evaluator.__new__(Evaluator)
        evaluator.batch_size = Config.BATCH_SIZE
        evaluator.inference_engine = MockInference()

        accuracies = []
        for run_idx in range(1, Config.REPEAT_N + 1):
            run_dir = os.path.join(Config.OUTPUT_DIR, f"run_{run_idx}")
            result = evaluator.evaluate(
                adaptor=adaptor, max_tokens=100,
                temperature=Config.TEMPERATURE, top_p=1.0,
                n_samples=Config.PASS_N, output_dir=run_dir
            )
            acc = result['report']['accuracy_percentage']
            accuracies.append(acc)

            pass_rates = result.get('per_question_pass_rates', [])
            for pr in pass_rates:
                assert 'pass_count' in pr, "missing pass_count"
                assert 'pass_rate' in pr, "missing pass_rate"
                assert 0 <= pr['pass_count'] <= 4, f"bad pass_count: {pr['pass_count']}"

        assert len(accuracies) == 3
        for a in accuracies:
            assert 0 <= a <= 100

        for run_idx in range(1, 4):
            pp = os.path.join(Config.OUTPUT_DIR, f"run_{run_idx}",
                              "per_question_pass_PASS4.jsonl")
            assert os.path.exists(pp), f"missing {pp}"
            with open(pp) as f:
                lines = [json.loads(l) for l in f if l.strip()]
            assert len(lines) == 8, f"expected 8 pass_rate entries, got {len(lines)}"

        print(f"PASS  (Pass@4: {[f'{a:.1f}%' for a in accuracies]})")
    finally:
        shutil.rmtree(tmp)


def test_repeat_disabled_when_temp_zero():
    """REPEAT_N=3, TEMPERATURE=0: repeat should NOT activate."""
    print("TEST: repeat disabled at T=0 ...", end=" ")
    from main import Config, _repeat_enabled
    Config.REPEAT_N = 3
    Config.PASS_N = 4
    Config.TEMPERATURE = 0
    assert not _repeat_enabled(), "should be disabled at T=0"
    print("PASS")


def test_no_repeat_when_repeat_n_is_1():
    """REPEAT_N=1: repeat should NOT activate regardless of other settings."""
    print("TEST: repeat disabled when REPEAT_N=1 ...", end=" ")
    from main import Config, _repeat_enabled
    Config.REPEAT_N = 1
    Config.PASS_N = 256
    Config.TEMPERATURE = 1.0
    assert not _repeat_enabled(), "should be disabled when REPEAT_N=1"
    print("PASS")


def test_summary_contains_passk_info():
    """_print_repeat_summary should include pass_n in output."""
    print("TEST: summary contains Pass@K ...", end=" ")
    from main import Config, _print_repeat_summary
    import io
    from contextlib import redirect_stdout

    tmp = tempfile.mkdtemp()
    try:
        Config.REPEAT_N = 2
        Config.PASS_N = 32

        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_repeat_summary([60.0, 70.0], tmp)

        out = buf.getvalue()
        assert "Pass@32" in out, f"'Pass@32' not in summary output:\n{out}"

        summary_path = os.path.join(tmp, "repeat_summary.json")
        assert os.path.exists(summary_path)
        with open(summary_path) as f:
            s = json.load(f)
        assert s['pass_n'] == 32
        assert s['metric'] == 'Pass@32'
        assert abs(s['mean_accuracy_percentage'] - 65.0) < 0.01

        print("PASS")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    print("=" * 60)
    print("Smoke Tests: REPEAT_N + Pass@K")
    print("=" * 60)

    test_repeat_disabled_when_temp_zero()
    test_no_repeat_when_repeat_n_is_1()
    test_repeat_pass1()
    test_repeat_passk()
    test_summary_contains_passk_info()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
