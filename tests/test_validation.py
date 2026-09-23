#!/usr/bin/env python3
"""CPU tests for length budget and best/final checkpoint selection."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opd_eval.length import (
    CORPUS_PASS16_MAX_NEW_TOKENS,
    MAX_MODEL_LEN,
    PROMPT_MAX_TOKENS,
    STUDENT_MAX_NEW_TOKENS,
    TEACHER_ACCEPT_MAX_TOKENS,
)
from opd_eval.roster import ROSTER, find_profile
from opd_eval.stats import StatsError, pool_from_records
from opd_eval.validation import (
    PRIMARY_METRIC,
    VAL_K,
    VAL_SURFACE_DT_TRAIN,
    VAL_SURFACE_HELD_OUT,
    is_saved_checkpoint_step,
    is_validation_step,
    load_metrics_jsonl,
    metrics_from_pool,
    parse_metrics_row,
    pass_at_1_from_pool,
    pass_at_8_from_pool,
    select_best_and_final,
    write_metrics_jsonl,
)


def _pool_records(bits_by_pid):
    recs = []
    for pid, bits in bits_by_pid.items():
        for idx, ok in enumerate(bits):
            recs.append({
                "problem_id": pid,
                "sample_idx": idx,
                "response": f"{pid}:{idx}",
                "verified": bool(ok),
                "n_tokens": 1,
            })
    return recs


class TestLengthBudget(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(PROMPT_MAX_TOKENS, 1024)
        self.assertEqual(STUDENT_MAX_NEW_TOKENS, 10240)
        self.assertEqual(MAX_MODEL_LEN, 11264)
        self.assertEqual(TEACHER_ACCEPT_MAX_TOKENS, 8192)
        self.assertEqual(CORPUS_PASS16_MAX_NEW_TOKENS, 10240)

    def test_main_config_matches(self):
        import importlib

        import main as main_mod

        importlib.reload(main_mod)
        Config = main_mod.Config

        self.assertEqual(Config.MAX_TOKENS, STUDENT_MAX_NEW_TOKENS)
        self.assertEqual(Config.MAX_MODEL_LEN, MAX_MODEL_LEN)
        self.assertEqual(Config.TEMPERATURE, 0.6)
        self.assertEqual(Config.TOP_P, 0.95)

    def test_roster_default_max_new_tokens(self):
        for p in ROSTER:
            self.assertEqual(p.max_new_tokens, STUDENT_MAX_NEW_TOKENS)

    def test_interface_documents_budget_and_u1_reuse(self):
        md = (Path(__file__).resolve().parents[1] / "INTERFACE.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("10240", md)
        self.assertIn("11264", md)
        self.assertIn("0.6", md)
        self.assertIn("0.95", md)
        self.assertIn("20260924", md)
        self.assertIn("heldout_h600", md)


class TestValidationMetrics(unittest.TestCase):
    def test_pass_at_1_and_pass_at_8(self):
        # p1: only idx0 true → pass@1 hit, pass@8 hit
        # p2: none true → both miss
        # p3: idx0 false, idx3 true → pass@1 miss, pass@8 hit
        bits = {
            "p1": (True, False, False, False, False, False, False, False),
            "p2": (False, False, False, False, False, False, False, False),
            "p3": (False, False, False, True, False, False, False, False),
        }
        pool = pool_from_records(_pool_records(bits), VAL_K)
        self.assertAlmostEqual(pass_at_1_from_pool(pool), 1.0 / 3.0)
        self.assertAlmostEqual(pass_at_8_from_pool(pool), 2.0 / 3.0)
        m = metrics_from_pool(pool)
        self.assertEqual(m["n_problems"], 3)
        self.assertEqual(m["k"], 8)
        self.assertEqual(m["g"], 0)
        self.assertEqual(m["max_new_tokens"], 10240)
        self.assertAlmostEqual(m["pass_at_1"], 1.0 / 3.0)
        self.assertAlmostEqual(m["pass_at_8"], 2.0 / 3.0)

    def test_rejects_nonzero_g(self):
        with self.assertRaises(StatsError):
            parse_metrics_row({
                "step": 100,
                "surface": VAL_SURFACE_HELD_OUT,
                "pass_at_1": 0.1,
                "pass_at_8": 0.2,
                "n_problems": 200,
                "g": 1,
            })


class TestBestFinalSelection(unittest.TestCase):
    def _rows(self):
        # Held-out every 50; saves at 100/200/300. Best pass@8 at 200.
        held = [
            {"step": 50, "surface": VAL_SURFACE_HELD_OUT,
             "pass_at_1": 0.10, "pass_at_8": 0.20, "n_problems": 200, "g": 0},
            {"step": 100, "surface": VAL_SURFACE_HELD_OUT,
             "pass_at_1": 0.12, "pass_at_8": 0.25, "n_problems": 200, "g": 0},
            {"step": 150, "surface": VAL_SURFACE_HELD_OUT,
             "pass_at_1": 0.15, "pass_at_8": 0.40, "n_problems": 200, "g": 0},
            {"step": 200, "surface": VAL_SURFACE_HELD_OUT,
             "pass_at_1": 0.18, "pass_at_8": 0.35, "n_problems": 200, "g": 0},
            {"step": 250, "surface": VAL_SURFACE_HELD_OUT,
             "pass_at_1": 0.17, "pass_at_8": 0.33, "n_problems": 200, "g": 0},
            {"step": 300, "surface": VAL_SURFACE_HELD_OUT,
             "pass_at_1": 0.16, "pass_at_8": 0.30, "n_problems": 200, "g": 0},
        ]
        # Train subset must not affect selection even if higher.
        train = [
            {"step": 200, "surface": VAL_SURFACE_DT_TRAIN,
             "pass_at_1": 0.99, "pass_at_8": 0.99, "n_problems": 64, "g": 0},
        ]
        return [parse_metrics_row(r) for r in held + train]

    def test_cadence_helpers(self):
        self.assertTrue(is_validation_step(50))
        self.assertTrue(is_validation_step(100))
        self.assertFalse(is_validation_step(75))
        self.assertTrue(is_saved_checkpoint_step(100))
        self.assertFalse(is_saved_checkpoint_step(50))
        self.assertFalse(is_saved_checkpoint_step(0))

    def test_selects_among_saved_only_and_ignores_train_surface(self):
        report = select_best_and_final(self._rows())
        # step 150 has higher pass@8 than 200 but is not a save → ignored
        self.assertEqual(report.best.step, 200)
        self.assertAlmostEqual(report.best.pass_at_8, 0.35)
        self.assertEqual(report.final.step, 300)
        self.assertEqual(report.primary_metric, PRIMARY_METRIC)
        self.assertEqual(
            [c.step for c in report.candidates],
            [100, 200, 300],
        )

    def test_tie_breaks_to_later_step(self):
        rows = [
            parse_metrics_row({
                "step": 100, "surface": VAL_SURFACE_HELD_OUT,
                "pass_at_1": 0.1, "pass_at_8": 0.5, "n_problems": 10, "g": 0,
            }),
            parse_metrics_row({
                "step": 200, "surface": VAL_SURFACE_HELD_OUT,
                "pass_at_1": 0.1, "pass_at_8": 0.5, "n_problems": 10, "g": 0,
            }),
        ]
        report = select_best_and_final(rows)
        self.assertEqual(report.best.step, 200)

    def test_explicit_final_step(self):
        report = select_best_and_final(self._rows(), final_step=200)
        self.assertEqual(report.final.step, 200)
        self.assertEqual(report.best.step, 200)

    def test_jsonl_roundtrip_and_cli_shape(self):
        rows = self._rows()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.jsonl"
            write_metrics_jsonl(path, rows)
            loaded = load_metrics_jsonl(path)
            self.assertEqual(len(loaded), len(rows))
            report = select_best_and_final(loaded)
            out = Path(tmp) / "selection.json"
            out.write_text(
                json.dumps(report.as_dict(), indent=2) + "\n", encoding="utf-8"
            )
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(data["best"]["step"], 200)
            self.assertEqual(data["final"]["step"], 300)
            self.assertIn("protocol", data)


class TestValidationRoster(unittest.TestCase):
    def test_validation_profiles_wired(self):
        held = find_profile(
            surface=VAL_SURFACE_HELD_OUT, benchmark_id="heldout_h", k=8
        )
        train = find_profile(
            surface=VAL_SURFACE_DT_TRAIN, benchmark_id="dt_train", k=8
        )
        paper = find_profile(surface="heldout_h", benchmark_id="heldout_h", k=128)
        self.assertEqual(held.k, 8)
        self.assertEqual(held.temperature, 0.6)
        self.assertEqual(held.top_p, 0.95)
        self.assertEqual(train.k, 8)
        self.assertEqual(paper.k, 128)
        self.assertNotEqual(held.output_dir("o", "C01", "slug", 100),
                            paper.output_dir("o", "C01", "slug", 100))


if __name__ == "__main__":
    unittest.main(verbosity=2)
