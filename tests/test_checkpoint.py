#!/usr/bin/env python3
"""Refusal path for INTERFACE.md §1: missing or nonzero g_eval is fatal."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opd_eval.checkpoint import (
    CheckpointRefused,
    read_checkpoint,
    resolve_eval_target,
)


VALID = {
    "run_id": "C01",
    "student_hf_id": "Qwen/Qwen3-1.7B-Base",
    "student_slug": "qwen3-1.7b-base",
    "teacher_hf_id": "Qwen/Qwen3-4B",
    "step": 1000,
    "g_eval": 0,
}


def _write_ckpt(root: Path, manifest: dict, *, make_hf: bool = True) -> Path:
    ckpt = root / "global_step_1000"
    ckpt.mkdir(parents=True)
    (ckpt / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if make_hf:
        (ckpt / "hf").mkdir()
        (ckpt / "hf" / "config.json").write_text("{}", encoding="utf-8")
    return ckpt


class TestCheckpointReader(unittest.TestCase):
    def test_accepts_g_eval_zero_and_loads_only_hf(self):
        with tempfile.TemporaryDirectory() as td:
            ckpt = _write_ckpt(Path(td), VALID)
            spec = read_checkpoint(ckpt)
            self.assertEqual(spec.g_eval, 0)
            self.assertEqual(spec.run_id, "C01")
            self.assertEqual(spec.step, 1000)
            self.assertEqual(spec.model_path, str(ckpt / "hf"))
            self.assertTrue(spec.model_path.endswith("/hf") or spec.model_path.endswith("\\hf"))
            self.assertFalse(spec.is_public_base)

    def test_refuses_missing_g_eval(self):
        manifest = dict(VALID)
        del manifest["g_eval"]
        with tempfile.TemporaryDirectory() as td:
            ckpt = _write_ckpt(Path(td), manifest)
            with self.assertRaises(CheckpointRefused) as ctx:
                read_checkpoint(ckpt)
            self.assertIn("g_eval", str(ctx.exception))
            self.assertIn("missing", str(ctx.exception))

    def test_refuses_nonzero_g_eval(self):
        for bad in (1, 0.5, "0", True):
            manifest = dict(VALID)
            manifest["g_eval"] = bad
            with tempfile.TemporaryDirectory() as td:
                ckpt = _write_ckpt(Path(td), manifest)
                with self.assertRaises(CheckpointRefused) as ctx:
                    read_checkpoint(ckpt)
                self.assertIn("g_eval", str(ctx.exception))

    def test_refuses_missing_hf_dir(self):
        with tempfile.TemporaryDirectory() as td:
            ckpt = _write_ckpt(Path(td), VALID, make_hf=False)
            with self.assertRaises(CheckpointRefused) as ctx:
                read_checkpoint(ckpt)
            self.assertIn("hf/", str(ctx.exception))

    def test_refuses_missing_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "global_step_1000"
            (ckpt / "hf").mkdir(parents=True)
            with self.assertRaises(CheckpointRefused):
                read_checkpoint(ckpt)

    def test_c00_bypasses_train_outputs(self):
        spec = resolve_eval_target("C00", "qwen3-1.7b-base", eval_seed=0)
        self.assertTrue(spec.is_public_base)
        self.assertEqual(spec.model_path, "Qwen/Qwen3-1.7B-Base")
        self.assertEqual(spec.g_eval, 0)
        spec_p = resolve_eval_target("C00p", "qwen3-0.6b-base", eval_seed=1)
        self.assertEqual(spec_p.model_path, "Qwen/Qwen3-0.6B-Base")
        self.assertEqual(spec_p.eval_seed, 1)

    def test_trained_run_without_ckpt_dir_refused(self):
        with self.assertRaises(CheckpointRefused):
            resolve_eval_target("C01", "qwen3-1.7b-base")


if __name__ == "__main__":
    unittest.main(verbosity=2)
