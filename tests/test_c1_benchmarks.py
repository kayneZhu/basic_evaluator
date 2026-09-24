#!/usr/bin/env python3
"""C.1 adaptors: roster wiring + prompt byte-identical to train render_g0_prompt."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptors.adaptor_factory import AdaptorFactory
from adaptors.c1_math_adaptor import (
    C1Aime24Adaptor,
    C1Amc23Adaptor,
    C1HeldoutHHardAdaptor,
    C1Hmmt25Adaptor,
    C1Math500Adaptor,
)
from adaptors.c1_prompt_mixin import render_c1_chatml
from opd_eval.roster import find_profile

EVAL_ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = EVAL_ROOT.parent / "train"
DATA = EVAL_ROOT / "data"


def _train_render(problem: str) -> str:
    """Load train ``prompt.py`` by path so fake ``torch`` stubs cannot poison it."""
    import importlib.util

    path = TRAIN_ROOT / "opd_frontier" / "data" / "prompt.py"
    spec = importlib.util.spec_from_file_location("_opd_train_c1_prompt", path)
    if spec is None or spec.loader is None:
        raise unittest.SkipTest(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.render_g0_prompt(problem, tokenizer=None)


class TestC1NewAdaptors(unittest.TestCase):
    def test_math500_amc_hmmt_sizes_and_ids(self):
        m = AdaptorFactory.create_adaptor(
            "c1_math500", str(DATA / "math500_bench_schema.jsonl")
        )
        self.assertIsInstance(m, C1Math500Adaptor)
        self.assertEqual(len(m.data), 500)
        self.assertTrue(m.get_problem_id(m.data[0]).startswith("math500:"))

        a = AdaptorFactory.create_adaptor(
            "c1_amc23", str(DATA / "amc23_bench_schema.jsonl")
        )
        self.assertIsInstance(a, C1Amc23Adaptor)
        self.assertEqual(len(a.data), 40)

        h = AdaptorFactory.create_adaptor(
            "c1_hmmt25", str(DATA / "hmmt25_bench_schema.jsonl")
        )
        self.assertIsInstance(h, C1Hmmt25Adaptor)
        self.assertEqual(len(h.data), 30)
        self.assertEqual(h.get_problem_id(h.data[0]), "hmmt25:1")

    def test_aime_year_filters_of_union(self):
        path = str(DATA / "aime24_25_26_bench_schema.jsonl")
        a24 = AdaptorFactory.create_adaptor("c1_aime24", path)
        a25 = AdaptorFactory.create_adaptor("c1_aime25", path)
        a26 = AdaptorFactory.create_adaptor("c1_aime26", path)
        self.assertIsInstance(a24, C1Aime24Adaptor)
        self.assertEqual(len(a24.data), 30)
        self.assertEqual(len(a25.data), 30)
        self.assertEqual(len(a26.data), 30)
        self.assertEqual(a24.get_problem_id(a24.data[0]), "aime24:0")
        self.assertEqual(a25.get_problem_id(a25.data[0]), "aime25:0")
        self.assertEqual(a26.get_problem_id(a26.data[0]), "aime26:0")

    def test_h_hard_filters_b0(self):
        path = str(DATA / "heldout_fixtures" / "h_fixture.jsonl")
        hard = AdaptorFactory.create_adaptor("c1_heldout_h_hard", path)
        self.assertIsInstance(hard, C1HeldoutHHardAdaptor)
        self.assertTrue(hard.data)
        self.assertTrue(all(str(r["bin"]) == "B0" for r in hard.data))

    def test_roster_entries(self):
        for surface, bid, key in [
            ("math500", "math500", "c1_math500"),
            ("amc23", "amc23", "c1_amc23"),
            ("hmmt25", "hmmt25", "c1_hmmt25"),
            ("heldout_h_hard", "heldout_h_hard", "c1_heldout_h_hard"),
            ("aime24", "aime24", "c1_aime24"),
            ("aime25", "aime25", "c1_aime25"),
            ("aime26", "aime26", "c1_aime26"),
        ]:
            p = find_profile(surface=surface, benchmark_id=bid)
            self.assertEqual(p.adaptor_key, key)

    def test_prompt_matches_train_one_sample_per_benchmark(self):
        if not (TRAIN_ROOT / "opd_frontier" / "data" / "prompt.py").is_file():
            self.skipTest("sibling train/ repo missing")
        cases = [
            ("c1_math500", DATA / "math500_bench_schema.jsonl"),
            ("c1_amc23", DATA / "amc23_bench_schema.jsonl"),
            ("c1_hmmt25", DATA / "hmmt25_bench_schema.jsonl"),
            ("c1_aime_union", DATA / "aime24_25_26_bench_schema.jsonl"),
            ("c1_aime24", DATA / "aime24_25_26_bench_schema.jsonl"),
            ("c1_heldout_h_hard", DATA / "heldout_fixtures" / "h_fixture.jsonl"),
        ]
        for key, path in cases:
            with self.subTest(key=key):
                adaptor = AdaptorFactory.create_adaptor(key, str(path))
                item = adaptor.data[0]
                problem = adaptor.get_question(item)
                self.assertEqual(
                    adaptor.format_prompt(item),
                    _train_render(problem),
                )
                self.assertEqual(
                    adaptor.format_prompt(item),
                    render_c1_chatml(problem),
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
