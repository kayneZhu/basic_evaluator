#!/usr/bin/env python3
"""Config-driven roster, C.1 factory keys, protocol-profile output paths."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptors.adaptor_factory import AdaptorFactory
from adaptors.c1_math_adaptor import (
    C1AimeUnionAdaptor,
    C1OR1200Adaptor,
    aime_union_problem_id,
    is_aime26_problem_id,
)
from adaptors.c1_prompt_mixin import C1_USER_SUFFIX, C1_SYSTEM_PROMPT
from adaptors.verl_aligned_adaptor import VerlAlignedAdaptor
from opd_eval.roster import ROSTER, ProtocolProfile, find_profile
from opd_eval.paths import samples_output_dir


EVAL_ROOT = Path(__file__).resolve().parents[1]
AIME_UNION = str(EVAL_ROOT / "data" / "aime24_25_26_bench_schema.jsonl")
OR1_200 = str(EVAL_ROOT / "data" / "ttn_test_200.jsonl")


class TestFactoryKeys(unittest.TestCase):
    def test_existing_verl_aligned_unchanged(self):
        adaptor = AdaptorFactory.create_adaptor("verl_aligned", OR1_200)
        self.assertIs(type(adaptor), VerlAlignedAdaptor)

    def test_c1_aime_union_registered(self):
        a = AdaptorFactory.create_adaptor("c1_aime_union", AIME_UNION)
        b = AdaptorFactory.create_adaptor("aime_union", AIME_UNION)
        self.assertIsInstance(a, C1AimeUnionAdaptor)
        self.assertIsInstance(b, C1AimeUnionAdaptor)
        self.assertTrue(a.raw_prompts)
        self.assertEqual(a._get_system_prompt(), C1_SYSTEM_PROMPT)

    def test_c1_or1_200_registered(self):
        a = AdaptorFactory.create_adaptor("c1_or1_200", OR1_200)
        self.assertIsInstance(a, C1OR1200Adaptor)


class TestAimeUnionAndOR1(unittest.TestCase):
    def test_aime_union_90_and_year_namespaces(self):
        adaptor = AdaptorFactory.create_adaptor("c1_aime_union", AIME_UNION)
        self.assertEqual(len(adaptor.data), 90)
        ids = [adaptor.get_problem_id(item) for item in adaptor.data]
        self.assertEqual(len(set(ids)), 90)
        self.assertEqual(ids[0], "aime24:0")
        self.assertEqual(ids[29], "aime24:29")
        self.assertEqual(ids[30], "aime25:0")
        self.assertEqual(ids[59], "aime25:29")
        self.assertEqual(ids[60], "aime26:0")
        self.assertEqual(ids[89], "aime26:29")
        self.assertEqual(sum(1 for i in ids if i.startswith("aime24:")), 30)
        self.assertEqual(sum(1 for i in ids if i.startswith("aime25:")), 30)
        self.assertEqual(sum(1 for i in ids if i.startswith("aime26:")), 30)
        self.assertTrue(is_aime26_problem_id("aime26:0"))
        self.assertFalse(is_aime26_problem_id("aime24:0"))
        self.assertEqual(
            aime_union_problem_id(adaptor.data[60]),
            "aime26:0",
        )

    def test_or1_200_uses_index_not_pass_count(self):
        adaptor = AdaptorFactory.create_adaptor("c1_or1_200", OR1_200)
        self.assertEqual(len(adaptor.data), 200)
        ids = [adaptor.get_problem_id(item) for item in adaptor.data]
        self.assertEqual(ids[0], f"or1:{adaptor.data[0]['index']}")
        self.assertEqual(len(set(ids)), 200)
        self.assertTrue(all(i.startswith("or1:") for i in ids))
        # Bins are not derived from the previous paper's pass@32 columns.
        meta = adaptor.get_variant_metadata(adaptor.data[0])
        self.assertEqual(set(meta), {"problem_id"})
        self.assertNotIn("pass_count", meta)
        self.assertNotIn("pass_rate", meta)
        self.assertIn("pass_count", adaptor.data[0])
        self.assertIn("pass_rate", adaptor.data[0])

    def test_c1_user_line_not_rho0(self):
        adaptor = AdaptorFactory.create_adaptor("c1_aime_union", AIME_UNION)
        rendered = adaptor.format_prompt(adaptor.data[0])
        self.assertTrue(rendered.endswith("<think>\n"))
        self.assertIn(C1_USER_SUFFIX, rendered)
        self.assertIn(adaptor.data[0]["question"], rendered)
        self.assertNotIn("Please reason step by step to solve this problem.", rendered)
        self.assertNotIn("Use this format:", rendered)


class TestRosterIsAConfigList(unittest.TestCase):
    def test_roster_is_not_a_hardcoded_nine(self):
        self.assertIsInstance(ROSTER, list)
        self.assertGreaterEqual(len(ROSTER), 2)
        # The list is the roster. Growing it is the API; nine is not a constant.
        extra = ProtocolProfile(
            surface="harm",
            benchmark_id="scibench",
            k=8,
            temperature=1.0,
            adaptor_key="c1_scibench",
            data_path="data/scibench_train.jsonl",
        )
        grown = list(ROSTER) + [extra]
        self.assertEqual(len(grown), len(ROSTER) + 1)

    def test_wired_surfaces(self):
        aime = find_profile(surface="aime_union", benchmark_id="aime_union")
        or1 = find_profile(surface="or1_200", benchmark_id="or1_200")
        self.assertEqual(aime.k, 512)
        self.assertEqual(aime.temperature, 1.0)
        self.assertEqual(aime.adaptor_key, "c1_aime_union")
        self.assertEqual(or1.k, 128)
        self.assertEqual(or1.adaptor_key, "c1_or1_200")
        for p in ROSTER:
            self.assertFalse(hasattr(p, "pass_count"))
            self.assertFalse(hasattr(p, "pass_rate"))
            self.assertNotIn("bin", p.__dataclass_fields__)

    def test_protocol_profile_prevents_disk_collision(self):
        nine = ProtocolProfile(
            surface="nine_bench",
            benchmark_id="scibench",
            k=8,
            temperature=1.0,
            adaptor_key="c1_scibench",
            data_path="data/scibench_train.jsonl",
        )
        harm = ProtocolProfile(
            surface="harm",
            benchmark_id="scibench",
            k=8,
            temperature=1.0,
            adaptor_key="c1_scibench",
            data_path="data/scibench_train.jsonl",
        )
        a = nine.output_dir("outputs", "C01", "qwen3-1.7b-base", 1000)
        b = harm.output_dir("outputs", "C01", "qwen3-1.7b-base", 1000)
        self.assertNotEqual(a, b)
        self.assertEqual(a.parent, b.parent)
        self.assertTrue(str(a).endswith("scibench/nine_bench_k8_t1"))
        self.assertTrue(str(b).endswith("scibench/harm_k8_t1"))
        # Same helper used by the writer.
        self.assertEqual(
            a,
            samples_output_dir(
                "outputs", "C01", "qwen3-1.7b-base", 1000,
                "scibench", "nine_bench", 8, 1.0,
            ),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
