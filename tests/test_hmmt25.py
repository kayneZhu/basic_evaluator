#!/usr/bin/env python3
"""HMMT'25 schema + Math-Verify cascade self-checks."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptors.adaptor_factory import AdaptorFactory

EVAL_ROOT = Path(__file__).resolve().parents[1]
HMMT = EVAL_ROOT / "data" / "hmmt25_bench_schema.jsonl"


class TestHmmt25Checker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adaptor = AdaptorFactory.create_adaptor("c1_hmmt25", str(HMMT))
        cls.assertEqual = unittest.TestCase.assertEqual
        if len(cls.adaptor.data) != 30:
            raise unittest.SkipTest(f"expected 30 HMMT rows, got {len(cls.adaptor.data)}")

    def test_gold_against_self_all_30(self):
        failures = []
        for item in self.adaptor.data:
            gt = self.adaptor.get_ground_truth(item)
            ok = self.adaptor.verify_answer(gt, gt)
            if not ok:
                failures.append((self.adaptor.get_problem_id(item), gt))
        self.assertEqual(
            failures,
            [],
            msg=f"gold≠gold failures: {failures}",
        )

    def test_reformatted_variants(self):
        """Spot-check intervals / tuples / fractions / expressions."""
        cases = [
            # (gold, variant) that should match
            ("103", "103"),
            ("103", "103.0"),
            (r"\frac{1}{576}", r"1/576"),
            (r"\frac{1}{576}", r"\dfrac{1}{576}"),
            ("3375", "3375"),
            (r"\left( 3, \frac{\pi}{2} \right)", r"(3, \pi/2)"),  # may fail
        ]
        # Prefer real HMMT golds for fraction/integer.
        fracs = [
            item for item in self.adaptor.data
            if "frac" in self.adaptor.get_ground_truth(item)
        ]
        ints = [
            item for item in self.adaptor.data
            if self.adaptor.get_ground_truth(item).isdigit()
        ]
        report = {"ok": [], "fail": []}
        variant_specs = []
        if ints:
            g = self.adaptor.get_ground_truth(ints[0])
            variant_specs.extend(
                [
                    (g, g),
                    (g, f"{g}.0"),
                    (g, f" {g} "),
                ]
            )
        if fracs:
            g = self.adaptor.get_ground_truth(fracs[0])
            variant_specs.extend(
                [
                    (g, g),
                    (g, g.replace(r"\frac", r"\dfrac")),
                ]
            )
            # plain a/b if gold is \frac{a}{b}
            if g.startswith(r"\frac"):
                # crude extract
                inner = g[len(r"\frac") :]
                variant_specs.append((g, inner.replace("{", "").replace("}", "/").rstrip("/")))
        # Synthetic tuple / interval / expression (not HMMT golds).
        variant_specs.extend(
            [
                (r"(1,2)", r"\left(1, 2\right)"),
                (r"[0,1]", r"[0, 1]"),
                (r"x^2+1", r"x^{2}+1"),
            ]
        )
        for gold, variant in variant_specs:
            ok = self.adaptor.verify_answer(variant, gold)
            bucket = "ok" if ok else "fail"
            report[bucket].append({"gold": gold, "variant": variant})
        # Persist failures for the owner report (assert all HMMT-derived ok).
        hmmt_fails = [
            r for r in report["fail"]
            if any(
                r["gold"] == self.adaptor.get_ground_truth(it)
                for it in self.adaptor.data
            )
        ]
        self.assertEqual(hmmt_fails, [], msg=f"HMMT variant failures: {hmmt_fails}")
        # Print full report for the ≤250w summary.
        print("\nHMMT checker variant report:", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
