"""CPU dry-run for the n=512 boundary runner, plus the pre-registered report."""

from __future__ import annotations

import gzip
import json
import os
import subprocess
import unittest
from pathlib import Path

from adaptors.adaptor_factory import AdaptorFactory
from adaptors.prompt_format import C2_ASSISTANT_PREFILL
from adaptors.verl_aligned_adaptor import _extract_boxed, _verify_math
from opd_eval.boundary512 import (
    C2_PREFILL_IDS,
    EVAL_ROOT,
    compact_row,
    main as boundary_main,
    prefill_boundary_ok,
    run_boundary,
)
from opd_eval.boundary_report import build_report, load_problems

DATA = EVAL_ROOT.parent / "docs" / "eng" / "reports" / "data" / "c2_base"


def _gz_rows(path: Path):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


class RenderAndGradeTests(unittest.TestCase):
    def setUp(self):
        self._env = os.environ.get("OPD_PROMPT_FORMAT")
        os.environ["OPD_PROMPT_FORMAT"] = "c2_nothink"

    def tearDown(self):
        if self._env is None:
            os.environ.pop("OPD_PROMPT_FORMAT", None)
        else:
            os.environ["OPD_PROMPT_FORMAT"] = self._env

    def test_prefill_ids_constant(self):
        self.assertEqual(C2_PREFILL_IDS, (151667, 271, 151668, 271))
        self.assertTrue(prefill_boundary_ok([1, 2, *C2_PREFILL_IDS]))
        self.assertFalse(prefill_boundary_ok([151667, 198, 151668, 271]))

    def test_two_problems_per_set_render_and_gold_box(self):
        union = str(EVAL_ROOT / "data" / "aime24_25_26_bench_schema.jsonl")
        amc = str(EVAL_ROOT / "data" / "amc23_bench_schema.jsonl")
        cases = [
            ("c1_aime24", union, "aime24:"),
            ("c1_aime25", union, "aime25:"),
            ("c1_aime26", union, "aime26:"),
            ("c1_amc23", amc, "amc23:"),
        ]
        for key, path, prefix in cases:
            adaptor = AdaptorFactory.create_adaptor(key, path)
            self.assertGreaterEqual(len(adaptor.data), 2)
            for item in adaptor.data[:2]:
                pid = adaptor.get_problem_id(item)
                self.assertTrue(pid.startswith(prefix), pid)
                prompt = adaptor.format_prompt(item)
                self.assertTrue(prompt.endswith(C2_ASSISTANT_PREFILL))
                self.assertIn("<|im_start|>system\n", prompt)
                gt = adaptor.get_ground_truth(item)
                boxed = f"work\n\\boxed{{{gt}}}"
                extracted = _extract_boxed(boxed)
                self.assertEqual(adaptor.extract_answer(boxed), extracted)
                self.assertTrue(adaptor.verify_answer(extracted, gt))
        self.assertTrue(_verify_math("204", "204"))
        self.assertTrue(_verify_math("204.0", "204"))
        self.assertTrue(_verify_math("-1", "-1"))
        self.assertTrue(_verify_math("3159", "3159"))
        self.assertFalse(_verify_math("26", "27"))
        self.assertFalse(_verify_math("", "27"))

    def test_compact_row_matches_boundary_schema(self):
        row = compact_row(
            {
                "problem_id": "aime24:0",
                "sample_idx": 3,
                "verified": True,
                "truncated": False,
                "n_tokens": 40,
                "terminal_token_id": 151643,
                "repetition": False,
            },
            "aime24",
        )
        self.assertEqual(
            set(row),
            {
                "problem_id",
                "bin",
                "sample_idx",
                "correct",
                "truncated",
                "n_tokens",
                "end_token_id",
                "repetition",
            },
        )
        self.assertEqual(row["end_token_id"], 151643)
        self.assertTrue(row["correct"])


class RunnerDryRunTests(unittest.TestCase):
    def setUp(self):
        self._env = os.environ.get("OPD_PROMPT_FORMAT")
        os.environ["OPD_PROMPT_FORMAT"] = "c2_nothink"

    def tearDown(self):
        if self._env is None:
            os.environ.pop("OPD_PROMPT_FORMAT", None)
        else:
            os.environ["OPD_PROMPT_FORMAT"] = self._env

    def test_dry_plan_and_strict_amc_gap(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            rc = boundary_main(
                [
                    "--out-dir",
                    str(out),
                    "--dry-plan",
                    "--sets",
                    "aime24,aime25,aime26,amc23",
                    "--n",
                    "512",
                    "--seed",
                    "20261005",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertFalse(out.exists())
            strict = boundary_main(
                [
                    "--out-dir",
                    str(out),
                    "--dry-plan",
                    "--strict-registered",
                    "--sets",
                    "aime24,amc23",
                ]
            )
            self.assertEqual(strict, 2)

    def test_cpu_stub_resume_grades_and_schema(self):
        import tempfile

        calls = {"n": 0}

        def generate(prompt: str, seed: int) -> str:
            del seed
            calls["n"] += 1
            if not hasattr(generate, "prompts"):
                generate.prompts = []
            generate.prompts.append(prompt)
            return r"\boxed{27}"

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            payload = run_boundary(
                out_dir=out,
                sets=["amc23", "aime26"],
                n=2,
                seed=20261005,
                chunk_problems=1,
                max_problems_per_set=2,
                generate_fn=generate,
            )
            self.assertEqual(payload["n_rows"], 8)
            self.assertTrue(generate.prompts[0].endswith(C2_ASSISTANT_PREFILL))
            self.assertIn("Cities", generate.prompts[0])
            rows = _gz_rows(out / "boundary.jsonl.gz")
            self.assertEqual(len(rows), 8)
            by_id = {}
            for row in rows:
                by_id.setdefault(row["problem_id"], []).append(row)
            self.assertTrue(all(row["correct"] for row in by_id["amc23:0"]))
            self.assertTrue(all(not row["correct"] for row in by_id["amc23:1"]))
            self.assertEqual(by_id["aime26:0"][0]["bin"], "aime26")
            self.assertIn("aime26:1", by_id)
            keys = [(row["problem_id"], row["sample_idx"]) for row in rows]
            self.assertEqual(len(keys), len(set(keys)))

            sample_path = out / "amc23_c000" / "samples.jsonl"
            kept = sample_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(kept), 2)
            sample_path.write_text(kept[0] + "\n", encoding="utf-8")
            (out / "chunks" / "amc23_c000.done").unlink()
            (out / "chunks" / "amc23_c000.jsonl.gz").unlink()
            (out / "COMPLETE").unlink()
            before = calls["n"]
            again = run_boundary(
                out_dir=out,
                sets=["amc23", "aime26"],
                n=2,
                seed=20261005,
                chunk_problems=1,
                max_problems_per_set=2,
                generate_fn=generate,
            )
            self.assertEqual(again["n_rows"], 8)
            self.assertEqual(calls["n"], before + 1)
            resumed = _gz_rows(out / "boundary.jsonl.gz")
            self.assertEqual(len(resumed), 8)
            self.assertEqual(
                len({(row["problem_id"], row["sample_idx"]) for row in resumed}),
                8,
            )


class ReportTests(unittest.TestCase):
    def test_synthetic_learned_forgotten_and_crossing(self):
        import tempfile

        def write(path: Path, rows):
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

        def expand(pid, bin_name, c, n=8):
            rows = []
            for idx in range(n):
                rows.append(
                    {
                        "problem_id": pid,
                        "bin": bin_name,
                        "sample_idx": idx,
                        "correct": idx < c,
                        "truncated": idx == 0,
                        "n_tokens": 10 + idx,
                        "end_token_id": 151643,
                        "repetition": False,
                    }
                )
            return rows

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = tmp / "base.jsonl.gz"
            model = tmp / "model.jsonl.gz"
            # c: base 0,2,8,0  model 1,0,8,0  across B0,B0,B1,B1
            write(
                base,
                expand("p0", "B0", 0)
                + expand("p1", "B0", 2)
                + expand("p2", "B1", 8)
                + expand("p3", "B1", 0),
            )
            write(
                model,
                expand("p0", "B0", 1)
                + expand("p1", "B0", 0)
                + expand("p2", "B1", 8)
                + expand("p3", "B1", 0),
            )
            report = build_report(base, {"m": model}, n_boot=40, seed=20261005)
            self.assertEqual(report["ks"], [1, 2, 4, 8])
            self.assertIn(16, report["ks_omitted"])
            pooled = report["vs_base"]["m"]["pooled"]
            self.assertEqual(pooled["learned"], 1)
            self.assertEqual(pooled["forgotten"], 1)
            self.assertEqual(pooled["base_solved"], 2)
            self.assertEqual(pooled["base_unsolved"], 2)
            self.assertEqual(pooled["crossing_k"], 1)
            self.assertEqual(report["base"]["pooled"]["solvable"], 2)
            self.assertEqual(report["models"]["m"]["B0"]["solvable"], 1)
            self.assertGreater(pooled["se"]["1"], 0)

    def test_n256_boundary_tables(self):
        base = DATA / "boundary_g1_base.jsonl.gz"
        if not base.is_file():
            self.skipTest("local n=256 boundary tables are not in this checkout")
        report = build_report(
            base,
            {
                "g2": DATA / "boundary_g2_s300.jsonl.gz",
                "g3": DATA / "boundary_g3_s300.jsonl.gz",
            },
            n_boot=2000,
            seed=20261005,
        )
        self.assertEqual(report["n"], 256)
        self.assertEqual(report["ks"][-1], 256)
        self.assertIn(512, report["ks_omitted"])
        self.assertEqual(report["base"]["B0"]["n_problems"], 50)
        self.assertEqual(report["base"]["B1"]["n_problems"], 50)
        self.assertEqual(report["base"]["B0"]["solvable"], 20)
        self.assertEqual(report["base"]["B1"]["solvable"], 49)
        self.assertEqual(report["base"]["pooled"]["solvable"], 69)
        self.assertEqual(report["models"]["g2"]["B0"]["solvable"], 23)
        self.assertEqual(report["models"]["g3"]["B0"]["solvable"], 26)
        self.assertEqual(report["models"]["g2"]["B1"]["solvable"], 49)
        self.assertEqual(report["models"]["g3"]["pooled"]["solvable"], 75)
        g2 = report["vs_base"]["g2"]
        g3 = report["vs_base"]["g3"]
        self.assertEqual(g2["B0"]["learned"], 7)
        self.assertEqual(g2["B0"]["forgotten"], 4)
        self.assertEqual(g2["B1"]["learned"], 0)
        self.assertEqual(g2["B1"]["forgotten"], 0)
        self.assertEqual(g3["B0"]["learned"], 9)
        self.assertEqual(g3["B0"]["forgotten"], 3)
        self.assertIsNone(g2["pooled"]["crossing_k"])
        self.assertIsNone(g3["pooled"]["crossing_k"])
        self.assertEqual(g2["B1"]["se"]["256"], 0.0)
        self.assertAlmostEqual(report["base"]["pooled"]["pass_at_k"]["1"], 0.0529, places=4)
        self.assertAlmostEqual(report["base"]["pooled"]["trunc_rate"], 0.2937, places=4)
        loaded = load_problems(base)
        self.assertEqual(len(loaded), 100)


class HostScriptTests(unittest.TestCase):
    def test_refuses_unset_prompt_format(self):
        script = EVAL_ROOT / "scripts" / "boundary512_host.sh"
        env = os.environ.copy()
        env.pop("OPD_PROMPT_FORMAT", None)
        proc = subprocess.run(
            ["bash", str(script), "/tmp/out", "base=/no/such"],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("c2_nothink", proc.stderr)


if __name__ == "__main__":
    unittest.main()
