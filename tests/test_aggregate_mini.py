#!/usr/bin/env python3
"""Unit tests for opd_eval.aggregate_mini (mini-protocol aggregation)."""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opd_eval.aggregate_mini import (
    ProblemRecord,
    aggregate_bench,
    cross_check_metrics_full,
    mean_unbiased_pass_at_k_from_nc,
    merge_b0_from_hard,
    pass_at_1_first16,
    pass_at_k_grid,
    run_model_dir,
    stream_samples_to_records,
    truncation_rate,
)
from opd_eval.stats import unbiased_pass_at_k


def _brute_pass_at_k(n: int, c: int, k: int) -> float:
    """Enumerate C(n,k) subsets; fraction that hit ≥1 success. Exact for small n."""
    if k == 0 or c == 0:
        return 0.0
    if k > n:
        raise ValueError("k>n")
    if n - c < k:
        return 1.0
    # successes occupy indices 0..c-1
    from itertools import combinations

    total = 0
    hit = 0
    for combo in combinations(range(n), k):
        total += 1
        if any(i < c for i in combo):
            hit += 1
    return hit / total


class TestPassAtKGrid(unittest.TestCase):
    def test_powers(self):
        self.assertEqual(pass_at_k_grid(16), [1, 2, 4, 8, 16])
        self.assertEqual(pass_at_k_grid(512)[:4], [1, 2, 4, 8])
        self.assertEqual(pass_at_k_grid(512)[-1], 512)
        self.assertEqual(pass_at_k_grid(40), [1, 2, 4, 8, 16, 32, 40])


class TestUnbiasedVsBrute(unittest.TestCase):
    def test_small_cases(self):
        for n, c, k in [
            (5, 2, 1),
            (5, 2, 2),
            (5, 2, 3),
            (6, 1, 2),
            (6, 3, 3),
            (7, 0, 2),
            (7, 7, 3),
            (8, 4, 4),
        ]:
            est = unbiased_pass_at_k(n, c, k)
            brute = _brute_pass_at_k(n, c, k)
            self.assertAlmostEqual(est, brute, places=12, msg=f"n={n} c={c} k={k}")


class TestDedupStream(unittest.TestCase):
    def test_dedup_keeps_first(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "samples.jsonl"
            rows = [
                {
                    "problem_id": "p1",
                    "sample_idx": 0,
                    "verified": True,
                    "n_tokens": 10,
                    "finish_reason": "stop",
                    "response": "a",
                },
                {
                    "problem_id": "p1",
                    "sample_idx": 0,  # dup
                    "verified": False,
                    "n_tokens": 99,
                    "finish_reason": "length",
                    "response": "b",
                },
                {
                    "problem_id": "p1",
                    "sample_idx": 1,
                    "verified": False,
                    "n_tokens": 20,
                    "finish_reason": "stop",
                    "response": "c",
                },
            ]
            p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            recs, meta = stream_samples_to_records(p, prefix_n=16)
            self.assertEqual(meta["n_dup_keys"], 1)
            self.assertEqual(len(recs), 1)
            self.assertEqual(recs[0].n, 2)
            self.assertEqual(recs[0].c, 1)  # first verified=True kept
            self.assertEqual(recs[0].n_first16, 2)
            self.assertEqual(recs[0].c_first16, 1)


class TestB0Reuse(unittest.TestCase):
    def test_reuse_missing_b0_from_hard_first16(self):
        mini = [
            ProblemRecord("a", n=16, c=4, bin="B1", n_first16=16, c_first16=4),
        ]
        hard = [
            ProblemRecord(
                "b0",
                n=512,
                c=10,
                bin="B0",
                n_first16=16,
                c_first16=2,
                mean_n_tokens=100.0,
                finish_reason_counts={"stop": 500, "length": 12},
            ),
        ]
        merged, meta = merge_b0_from_hard(mini, hard, b0_ids={"b0"})
        self.assertEqual(meta["reused"], 1)
        by = {r.problem_id: r for r in merged}
        self.assertEqual(by["b0"].n, 16)
        self.assertEqual(by["b0"].c, 2)
        self.assertEqual(by["b0"].bin, "B0")

    def test_keep_existing_b0(self):
        mini = [
            ProblemRecord("b0", n=16, c=3, bin="B0", n_first16=16, c_first16=3),
        ]
        hard = [
            ProblemRecord("b0", n=512, c=0, bin="B0", n_first16=16, c_first16=0),
        ]
        merged, meta = merge_b0_from_hard(mini, hard, b0_ids={"b0"})
        self.assertEqual(meta["reused"], 0)
        self.assertEqual(merged[0].c, 3)


class TestPerBinAgg(unittest.TestCase):
    def test_per_bin(self):
        recs = [
            ProblemRecord("p0", n=16, c=0, bin="B0", n_first16=16, c_first16=0,
                          finish_reason_counts={"stop": 16}, mean_n_tokens=10),
            ProblemRecord("p1", n=16, c=8, bin="B1", n_first16=16, c_first16=8,
                          finish_reason_counts={"stop": 14, "length": 2}, mean_n_tokens=20),
            ProblemRecord("p2", n=16, c=16, bin="B1", n_first16=16, c_first16=16,
                          finish_reason_counts={"stop": 16}, mean_n_tokens=30),
        ]
        agg = aggregate_bench("h_mini", recs, expect_n=16)
        self.assertIn("B0", agg.by_bin)
        self.assertIn("B1", agg.by_bin)
        self.assertEqual(agg.by_bin["B0"]["n_problems"], 1)
        self.assertEqual(agg.by_bin["B1"]["n_problems"], 2)
        # B1 mean pass@1 = mean(8/16, 16/16) = 0.75
        self.assertAlmostEqual(agg.by_bin["B1"]["pass_at_1_pool"], 0.75)
        self.assertAlmostEqual(truncation_rate(recs), 2 / 48)


class TestEndToEndFixture(unittest.TestCase):
    def test_tiny_model_dir(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # two benches
            for bench, n, rows in [
                (
                    "amc23",
                    16,
                    [
                        {"problem_id": "a1", "n": 16, "c": 4, "n_first16": 16, "c_first16": 4,
                         "mean_n_tokens": 100, "finish_reason_counts": {"stop": 15, "length": 1}},
                        {"problem_id": "a2", "n": 16, "c": 0, "n_first16": 16, "c_first16": 0,
                         "mean_n_tokens": 200, "finish_reason_counts": {"stop": 16}},
                    ],
                ),
                (
                    "h_mini",
                    16,
                    [
                        {"problem_id": "h0", "bin": "B0", "n": 16, "c": 1, "n_first16": 16, "c_first16": 1,
                         "mean_n_tokens": 50, "finish_reason_counts": {"stop": 16}},
                        {"problem_id": "h1", "bin": "B3", "n": 16, "c": 16, "n_first16": 16, "c_first16": 16,
                         "mean_n_tokens": 40, "finish_reason_counts": {"stop": 16}},
                    ],
                ),
            ]:
                d = root / bench
                d.mkdir()
                with open(d / f"{bench}_per_problem.jsonl", "w", encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps(r) + "\n")
                # metrics.json matching pool pass@1
                pool_p1 = mean_unbiased_pass_at_k_from_nc(
                    [ProblemRecord(r["problem_id"], r["n"], r["c"]) for r in rows], 1
                )
                (d / "metrics.json").write_text(
                    json.dumps(
                        {
                            "n": n,
                            "n_problems": len(rows),
                            "mean_pass_at_1": pool_p1,
                            "mean_pass_at_k": {
                                "1": pool_p1,
                                str(n): mean_unbiased_pass_at_k_from_nc(
                                    [ProblemRecord(r["problem_id"], r["n"], r["c"]) for r in rows],
                                    n,
                                ),
                            },
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )

            out = run_model_dir(
                root,
                size="1.7B",
                method="Base",
                checkpoint="base",
                benches=["amc23", "h_mini"],
            )
            self.assertTrue(out["cross_check"]["amc23"]["ok"])
            self.assertTrue(out["cross_check"]["h_mini"]["ok"])
            self.assertTrue((Path(out["out_dir"]) / "main_table.csv").is_file())
            self.assertTrue((Path(out["out_dir"]) / "figure_pass_at_k.csv").is_file())
            self.assertTrue((Path(out["out_dir"]) / "per_bin_h.csv").is_file())

            # first-16 pass@1 for amc23 = mean(4/16, 0) = 0.125
            amc = next(s for s in out["summaries"] if s["benchmark"] == "amc23")
            self.assertAlmostEqual(amc["pass_at_1_main"], 0.125)
            self.assertAlmostEqual(amc["pass_at_1_pool"], 0.125)


class TestCrossCheckDiff(unittest.TestCase):
    def test_detects_mismatch(self):
        recs = [ProblemRecord("p", n=16, c=4)]
        metrics = {"mean_pass_at_1": 0.5, "mean_pass_at_k": {"1": 0.5}, "n_problems": 1}
        diffs = cross_check_metrics_full(recs, metrics)
        self.assertTrue(any(d["field"] == "mean_pass_at_1" for d in diffs))


if __name__ == "__main__":
    unittest.main()
