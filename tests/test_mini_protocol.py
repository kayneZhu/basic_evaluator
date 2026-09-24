#!/usr/bin/env python3
"""Chen unbiased pass@k + mini-set builder + seeds + mini protocol resume."""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptors.adaptor_factory import AdaptorFactory
from opd_eval.mini_protocol import (
    plan_work_items,
    run_job,
    MiniJob,
)
from opd_eval.mini_sets import (
    build_h_hard_mini,
    build_h_mini,
    build_math500_mini,
    h_minus_hs,
    read_jsonl,
    write_jsonl,
)
from opd_eval.sampling import (
    BINNING_SEED,
    EVAL_SEED_MINI_PROTOCOL,
    refuse_binning_seed,
    sample_seed,
)
from opd_eval.stats import (
    StatsError,
    mean_pass_at_k,
    mean_unbiased_pass_at_k,
    unbiased_pass_at_k,
    unbiased_pass_at_k_curve,
)

EVAL_ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = EVAL_ROOT.parent / "train"
FIXTURE_H = EVAL_ROOT / "data" / "heldout_fixtures" / "h_fixture.jsonl"
FIXTURE_HS = EVAL_ROOT / "data" / "heldout_fixtures" / "hs_fixture.jsonl"
MATH500 = EVAL_ROOT / "data" / "math500_bench_schema.jsonl"


class TestUnbiasedPassAtK(unittest.TestCase):
    def test_hand_values(self):
        # n=5, c=2, k=1 → 2/5
        self.assertAlmostEqual(unbiased_pass_at_k(5, 2, 1), 2 / 5)
        # k=2 → 1 - C(3,2)/C(5,2) = 1 - 3/10 = 0.7
        self.assertAlmostEqual(unbiased_pass_at_k(5, 2, 2), 0.7)
        self.assertEqual(unbiased_pass_at_k(5, 0, 3), 0.0)
        self.assertEqual(unbiased_pass_at_k(5, 5, 3), 1.0)
        self.assertEqual(unbiased_pass_at_k(5, 2, 4), 1.0)  # n-c < k

    def test_matches_comb_when_small(self):
        n, c, k = 10, 3, 4
        expected = 1.0 - math.comb(n - c, k) / math.comb(n, k)
        self.assertAlmostEqual(unbiased_pass_at_k(n, c, k), expected)

    def test_curve_and_mean(self):
        curve = unbiased_pass_at_k_curve(4, 1)
        self.assertEqual(set(curve), {1, 2, 3, 4})
        pool = {
            "a": (True, False, False, False),
            "b": (False, False, False, False),
        }
        self.assertAlmostEqual(mean_unbiased_pass_at_k(pool, 4, 1), 0.125)
        # legacy raw still available
        self.assertAlmostEqual(mean_pass_at_k(pool, 4), 0.5)

    def test_rejects_illegal(self):
        with self.assertRaises(StatsError):
            unbiased_pass_at_k(3, 4, 1)


class TestSeeds(unittest.TestCase):
    def test_mini_seed_constant(self):
        self.assertEqual(EVAL_SEED_MINI_PROTOCOL, 20261001)
        self.assertNotEqual(EVAL_SEED_MINI_PROTOCOL, BINNING_SEED)
        self.assertNotIn(EVAL_SEED_MINI_PROTOCOL, {0, 1})

    def test_refuse_binning_seed(self):
        with self.assertRaises(ValueError):
            refuse_binning_seed(BINNING_SEED)
        with self.assertRaises(ValueError):
            sample_seed(BINNING_SEED, "or1:1", 0)
        refuse_binning_seed(EVAL_SEED_MINI_PROTOCOL)

    def test_sample_seed_matches_train(self):
        path = TRAIN_ROOT / "opd_frontier" / "data" / "rollout_passk.py"
        if not path.is_file():
            self.skipTest("sibling train/ missing")
        # Avoid importing the train package (vLLM / torch stub collisions).
        src = path.read_text(encoding="utf-8")
        self.assertIn('key = f"{base_seed}|{or1_id}|passk|{sample_idx}"', src)
        self.assertIn("h = (h * 131 + ord(ch)) & 0x7FFFFFFF", src)
        for pid, idx in [("or1:1", 0), ("math500:x", 7), ("aime24:0", 511)]:
            key = f"{EVAL_SEED_MINI_PROTOCOL}|{pid}|passk|{idx}"
            h = 0
            for ch in key:
                h = (h * 131 + ord(ch)) & 0x7FFFFFFF
            self.assertEqual(sample_seed(EVAL_SEED_MINI_PROTOCOL, pid, idx), h)


class TestMiniSets(unittest.TestCase):
    def test_h_minus_hs_and_stratified(self):
        h = read_jsonl(FIXTURE_H)
        hs = read_jsonl(FIXTURE_HS)
        rem = h_minus_hs(h, hs)
        rem_ids = {r["problem_id"] for r in rem}
        for r in hs:
            self.assertNotIn(r["problem_id"], rem_ids)
        # fixture: 3/bin H, 1/bin Hs → 2/bin remainder
        mini = build_h_mini(h, hs, per_bin=2, seed=20260924)
        self.assertEqual(len(mini), 8)
        from collections import Counter

        self.assertEqual(
            Counter(r["bin"] for r in mini),
            Counter({"B0": 2, "B1": 2, "B2": 2, "B3": 2}),
        )

    def test_h_hard_mini_prefix(self):
        h = read_jsonl(FIXTURE_H)
        hs = read_jsonl(FIXTURE_HS)
        # Need larger synthetic B0 pool for size=80 — build local synthetic.
        b0 = []
        for i in range(100):
            b0.append(
                {
                    "problem_id": f"or1:{i}",
                    "or1_id": f"or1:{i}",
                    "bin": "B0",
                    "question": "q",
                    "ground_truth": "1",
                    "index": i,
                }
            )
        hs_ids = {f"or1:{i}" for i in range(10)}
        hs_rows = [r for r in b0 if r["problem_id"] in hs_ids]
        h_rows = b0
        for b in ("B1", "B2", "B3"):
            for i in range(20):
                h_rows = h_rows  # B0 only needed for hard-mini
        mini = build_h_mini(
            h_rows + [
                {"problem_id": f"or1:b1_{i}", "or1_id": f"or1:b1_{i}", "bin": "B1", "question": "q", "ground_truth": "1", "index": 1000 + i}
                for i in range(30)
            ] + [
                {"problem_id": f"or1:b2_{i}", "or1_id": f"or1:b2_{i}", "bin": "B2", "question": "q", "ground_truth": "1", "index": 2000 + i}
                for i in range(30)
            ] + [
                {"problem_id": f"or1:b3_{i}", "or1_id": f"or1:b3_{i}", "bin": "B3", "question": "q", "ground_truth": "1", "index": 3000 + i}
                for i in range(30)
            ],
            hs_rows,
            per_bin=5,
            seed=7,
        )
        hard = build_h_hard_mini(
            h_rows + [
                {"problem_id": f"or1:b1_{i}", "or1_id": f"or1:b1_{i}", "bin": "B1", "question": "q", "ground_truth": "1", "index": 1000 + i}
                for i in range(30)
            ] + [
                {"problem_id": f"or1:b2_{i}", "or1_id": f"or1:b2_{i}", "bin": "B2", "question": "q", "ground_truth": "1", "index": 2000 + i}
                for i in range(30)
            ] + [
                {"problem_id": f"or1:b3_{i}", "or1_id": f"or1:b3_{i}", "bin": "B3", "question": "q", "ground_truth": "1", "index": 3000 + i}
                for i in range(30)
            ],
            hs_rows,
            mini,
            size=80,
            seed=7,
        )
        self.assertEqual(len(hard), 80)
        prefix = [r for r in mini if r["bin"] == "B0"]
        self.assertEqual(
            [r["problem_id"] for r in hard[: len(prefix)]],
            [r["problem_id"] for r in prefix],
        )

    def test_math500_mini_deterministic(self):
        rows = read_jsonl(MATH500)
        a = build_math500_mini(rows, size=200, seed=20260924)
        b = build_math500_mini(rows, size=200, seed=20260924)
        self.assertEqual(len(a), 200)
        self.assertEqual(
            [r["unique_id"] for r in a],
            [r["unique_id"] for r in b],
        )
        c = build_math500_mini(rows, size=200, seed=20260925)
        self.assertNotEqual(
            [r["unique_id"] for r in a],
            [r["unique_id"] for r in c],
        )

    def test_does_not_materialize_h_mini_by_default(self):
        # Script API: omitting --h-mini-per-bin writes nothing for h_mini.
        from opd_eval.mini_sets import main as mini_main

        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            rc = mini_main(
                [
                    "--heldout-h",
                    str(FIXTURE_H),
                    "--hs",
                    str(FIXTURE_HS),
                    "--out-dir",
                    str(out),
                    "--write-math500-mini",
                    "--math500",
                    str(MATH500),
                ]
            )
            self.assertEqual(rc, 0)
            self.assertFalse((out / "h_mini.jsonl").exists())
            self.assertTrue((out / "math500_mini.jsonl").is_file())


class TestMiniProtocolResume(unittest.TestCase):
    def test_plan_skips_done_and_refuses_binning_seed(self):
        path = str(EVAL_ROOT / "data" / "amc23_bench_schema.jsonl")
        adaptor = AdaptorFactory.create_adaptor("c1_amc23", path)
        done = {(adaptor.get_problem_id(adaptor.data[0]), 0)}
        work = plan_work_items(
            adaptor, n=2, base_seed=EVAL_SEED_MINI_PROTOCOL, done=done
        )
        self.assertNotIn(
            (adaptor.get_problem_id(adaptor.data[0]), 0),
            {(w["problem_id"], w["sample_idx"]) for w in work},
        )
        self.assertEqual(len(work), 40 * 2 - 1)
        with self.assertRaises(ValueError):
            plan_work_items(adaptor, n=1, base_seed=BINNING_SEED, done=set())

    def test_generate_fn_resume_and_metrics(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # tiny amc slice
            data = root / "amc.jsonl"
            rows = read_jsonl(EVAL_ROOT / "data" / "amc23_bench_schema.jsonl")[:2]
            write_jsonl(data, rows)
            job = MiniJob(
                surface="amc23",
                benchmark_id="amc23",
                adaptor_key="c1_amc23",
                data_path=data,
                n=2,
            )

            def gen(prompt: str, seed: int) -> str:
                # Always box the ground truth digit if present in prompt — use seed parity
                return r"\boxed{27}" if seed % 2 == 0 else r"\boxed{0}"

            info = run_job(
                job,
                out_root=root,
                model_dir=root / "hf",
                base_seed=EVAL_SEED_MINI_PROTOCOL,
                generate_fn=gen,
            )
            self.assertEqual(info["n_todo"], 4)
            samples = root / "amc23" / "samples.jsonl"
            self.assertTrue(samples.is_file())
            # resume: second call should todo 0
            info2 = run_job(
                job,
                out_root=root,
                model_dir=root / "hf",
                base_seed=EVAL_SEED_MINI_PROTOCOL,
                generate_fn=gen,
            )
            self.assertEqual(info2["n_todo"], 0)
            self.assertTrue((root / "amc23" / "metrics.json").is_file())
            self.assertTrue((root / "amc23" / "per_problem_nc.jsonl").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
