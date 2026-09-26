"""CPU tests for mini_protocol worker teardown / GPU idle helpers."""

from __future__ import annotations

import os
import signal
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opd_eval import mini_protocol as mp


def _write_mock_nvidia_smi(bin_dir: Path, script: str) -> Path:
    path = bin_dir / "nvidia-smi"
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


class TestGpuIdleHelpers(unittest.TestCase):
    def test_gpus_are_idle_true(self):
        with tempfile.TemporaryDirectory() as td:
            bin_dir = Path(td)
            smi = _write_mock_nvidia_smi(
                bin_dir,
                """#!/usr/bin/env bash
if [[ "$*" == *memory.used* ]]; then
  echo "12"
  echo "8"
  exit 0
fi
if [[ "$*" == *compute-apps* ]]; then
  exit 0
fi
exit 0
""",
            )
            self.assertTrue(mp.gpus_are_idle(nvidia_smi=str(smi)))
            self.assertEqual(mp.gpu_memory_used_mib(nvidia_smi=str(smi)), [12, 8])
            self.assertEqual(mp.gpu_compute_app_pids(nvidia_smi=str(smi)), [])

    def test_gpus_are_idle_false_high_mem(self):
        with tempfile.TemporaryDirectory() as td:
            bin_dir = Path(td)
            smi = _write_mock_nvidia_smi(
                bin_dir,
                """#!/usr/bin/env bash
if [[ "$*" == *memory.used* ]]; then
  echo "1200"
  echo "8"
  exit 0
fi
exit 0
""",
            )
            self.assertFalse(mp.gpus_are_idle(nvidia_smi=str(smi), mem_mib_limit=1000))

    def test_gpus_are_idle_false_compute_apps(self):
        with tempfile.TemporaryDirectory() as td:
            bin_dir = Path(td)
            smi = _write_mock_nvidia_smi(
                bin_dir,
                """#!/usr/bin/env bash
if [[ "$*" == *memory.used* ]]; then
  echo "12"
  exit 0
fi
if [[ "$*" == *compute-apps* ]]; then
  echo "12345"
  exit 0
fi
exit 0
""",
            )
            self.assertFalse(mp.gpus_are_idle(nvidia_smi=str(smi)))
            self.assertEqual(mp.gpu_compute_app_pids(nvidia_smi=str(smi)), [12345])

    def test_wait_for_gpus_idle_polls_then_succeeds(self):
        sleeps: list[float] = []
        calls = {"n": 0}

        def fake_idle(*, mem_mib_limit=1000, nvidia_smi="nvidia-smi"):
            calls["n"] += 1
            return calls["n"] >= 3

        with mock.patch.object(mp, "gpus_are_idle", side_effect=fake_idle):
            with mock.patch.object(mp, "gpu_memory_used_mib", return_value=[10]):
                ok = mp.wait_for_gpus_idle(
                    timeout_s=10,
                    poll_s=1,
                    sleep_fn=lambda s: sleeps.append(s),
                )
        self.assertTrue(ok)
        self.assertEqual(len(sleeps), 2)

    def test_wait_for_gpus_idle_timeout(self):
        sleeps: list[float] = []
        with mock.patch.object(mp, "gpus_are_idle", return_value=False):
            with mock.patch.object(mp, "gpu_memory_used_mib", return_value=[9000]):
                with mock.patch.object(mp, "gpu_compute_app_pids", return_value=[1]):
                    ok = mp.wait_for_gpus_idle(
                        timeout_s=2,
                        poll_s=1,
                        sleep_fn=lambda s: sleeps.append(s),
                    )
        self.assertFalse(ok)
        self.assertGreaterEqual(len(sleeps), 2)


class TestTerminateWorkers(unittest.TestCase):
    def test_terminate_sends_killpg(self):
        procs = []
        for _ in range(2):
            p = mock.Mock(spec=subprocess.Popen)
            p.pid = 4242
            p.poll.return_value = None
            p.returncode = None
            p.wait.return_value = 0
            procs.append(p)

        killpg_calls: list[tuple[int, int]] = []

        def fake_killpg(pid, sig):
            killpg_calls.append((pid, sig))
            for p in procs:
                p.poll.return_value = -sig

        with mock.patch.object(mp.os, "killpg", side_effect=fake_killpg):
            mp.terminate_worker_processes(procs, grace_s=0, sleep_fn=lambda _s: None)

        # TERM then KILL for each live proc (may skip KILL if poll says dead).
        self.assertTrue(any(sig == signal.SIGTERM for _, sig in killpg_calls))
        for p in procs:
            p.wait.assert_called()


class TestRetireMergedShards(unittest.TestCase):
    def test_merge_retire_rotates_and_truncates(self):
        with tempfile.TemporaryDirectory() as td:
            bench = Path(td)
            samples = bench / "samples.jsonl"
            shard = bench / "_gpu0_samples.jsonl"
            row = {
                "problem_id": "p1",
                "sample_idx": 0,
                "correct": True,
            }
            shard.write_text(
                __import__("json").dumps(row) + "\n", encoding="utf-8"
            )
            added = mp.merge_shard_samples(bench, samples, retire_shards=True)
            self.assertEqual(added, 1)
            self.assertTrue(samples.is_file())
            self.assertEqual(shard.read_text(encoding="utf-8"), "")
            rotated = bench / "_gpu0_samples.jsonl.merged"
            self.assertTrue(rotated.is_file())
            self.assertIn("p1", rotated.read_text(encoding="utf-8"))


class TestRunVllmDpTeardown(unittest.TestCase):
    def test_start_new_session_and_kill_siblings_on_failure(self):
        work = [
            {
                "problem_id": f"p{i}",
                "sample_idx": 0,
                "seed": i,
                "prompt": "x",
                "ground_truth": "y",
            }
            for i in range(2)
        ]
        pops: list[mock.Mock] = []

        def fake_popen(cmd, env=None, start_new_session=False):
            self.assertTrue(start_new_session)
            p = mock.Mock()
            p.pid = 1000 + len(pops)
            # First worker fails immediately; second still running.
            if not pops:
                p.poll.side_effect = [1]
                p.returncode = 1
            else:
                p.poll.side_effect = [None, None, None, -9]
                p.returncode = -9
            p.wait.return_value = p.returncode
            pops.append(p)
            return p

        with tempfile.TemporaryDirectory() as td:
            bench = Path(td)
            with mock.patch.object(mp.subprocess, "Popen", side_effect=fake_popen):
                with mock.patch.object(mp, "terminate_worker_processes") as term:
                    with mock.patch.object(mp, "wait_for_gpus_idle", return_value=True):
                        with mock.patch.object(mp, "write_progress"):
                            with mock.patch.object(mp, "progress_n_done", return_value=0):
                                with self.assertRaises(RuntimeError):
                                    mp._run_vllm_dp(
                                        work,
                                        bench_dir=bench,
                                        benchmark_id="t",
                                        adaptor_key="k",
                                        data_path="/tmp/x.jsonl",
                                        model_dir=Path("/tmp/model"),
                                        n_gpus=2,
                                        max_num_seqs=1,
                                        max_new_tokens=8,
                                        max_model_len=32,
                                        temperature=0.6,
                                        top_p=0.95,
                                        gpu_memory_utilization=0.9,
                                        gpu_ids="0,1",
                                    )
                self.assertTrue(term.called)
                self.assertEqual(len(pops), 2)


if __name__ == "__main__":
    unittest.main()
