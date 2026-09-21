#!/usr/bin/env python3
"""
CPU-only tests for global sample_idx, shard write/resume, and merge.

avg@8 is sample_idx 0–7 of the pooled file; p̂ uses the pooled K. A shard
that remaps 64–127 → 0..63 would still look well-formed and silently
corrupt both — these tests refuse that.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opd_eval.samples import (
    MergeIncompleteError,
    ShardConflictError,
    ShardError,
    ShardManifest,
    ShardManifestError,
    dumps_record,
    existing_keys,
    merge_records,
    merge_shards,
    samples_jsonl_path,
    split_sample_axis,
    work_items,
    write_shard,
)


PROBLEMS = ("pA", "pB", "pC")
K = 8


def _rec(pid: str, idx: int, *, verified: bool = False, suffix: str = "") -> dict:
    return {
        "problem_id": pid,
        "sample_idx": idx,
        "response": f"{pid}:{idx}{suffix}",
        "verified": verified,
        "n_tokens": 10 + idx,
    }


def _full_pool(k: int = K, problems=PROBLEMS) -> list:
    return [
        _rec(pid, idx, verified=((hash(pid) + idx) % 3 == 0))
        for pid in problems
        for idx in range(k)
    ]


def _write_sample_axis_partition(root: Path, records, n_shards: int, k: int, problems) -> bytes:
    ranges = split_sample_axis(k, n_shards)
    for shard_idx, (start, end) in enumerate(ranges):
        owned = [r for r in records if start <= r["sample_idx"] < end]
        manifest = ShardManifest(
            shard_idx=shard_idx,
            shard_count=n_shards,
            sample_idx_start=start,
            sample_idx_end=end,
            problem_ids=None,
        )
        write_shard(root, manifest, owned)
    return merge_shards(root, shard_count=n_shards, k=k, problem_ids=problems)


class TestSampleAxisPartition(unittest.TestCase):
    def test_ranges_are_global_and_cover_k(self):
        ranges = split_sample_axis(512, 8)
        self.assertEqual(ranges, [(i * 64, (i + 1) * 64) for i in range(8)])
        self.assertEqual(ranges[1], (64, 128))
        covered = [idx for start, end in ranges for idx in range(start, end)]
        self.assertEqual(covered, list(range(512)))

    def test_work_items_keep_global_sample_idx(self):
        items = work_items(["x"], 64, 68)
        self.assertEqual(items, [("x", 64), ("x", 65), ("x", 66), ("x", 67)])
        self.assertNotIn(("x", 0), items)

    def test_write_rejects_local_renumber(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = ShardManifest(
                shard_idx=1,
                shard_count=8,
                sample_idx_start=64,
                sample_idx_end=128,
                problem_ids=None,
            )
            with self.assertRaises(ShardError):
                write_shard(root, manifest, [_rec("pA", 0)])


class TestPartitionInvariance(unittest.TestCase):
    def test_merge_byte_identical_for_1_3_7_shards(self):
        records = _full_pool()
        payloads = []
        for n in (1, 3, 7):
            with tempfile.TemporaryDirectory() as td:
                payloads.append(
                    _write_sample_axis_partition(Path(td), records, n, K, PROBLEMS)
                )
        self.assertEqual(payloads[0], payloads[1])
        self.assertEqual(payloads[0], payloads[2])
        # Canonical sort and required fields.
        lines = payloads[0].decode("utf-8").splitlines()
        self.assertEqual(len(lines), len(PROBLEMS) * K)
        keys = []
        for line in lines:
            rec = __import__("json").loads(line)
            keys.append((rec["problem_id"], rec["sample_idx"]))
            self.assertEqual(
                list(rec.keys())[:5],
                ["problem_id", "sample_idx", "response", "verified", "n_tokens"],
            )
        self.assertEqual(
            keys,
            [(pid, idx) for pid in sorted(PROBLEMS) for idx in range(K)],
        )

    def test_mixed_problem_and_sample_rectangle_matches(self):
        records = _full_pool()
        with tempfile.TemporaryDirectory() as td_a, tempfile.TemporaryDirectory() as td_b:
            a = _write_sample_axis_partition(Path(td_a), records, 1, K, PROBLEMS)
            # Two shards: (all problems, idx 0-4) and (all problems, idx 4-8).
            root = Path(td_b)
            write_shard(
                root,
                ShardManifest(0, 2, 0, 4, None),
                [r for r in records if r["sample_idx"] < 4],
            )
            write_shard(
                root,
                ShardManifest(1, 2, 4, 8, None),
                [r for r in records if r["sample_idx"] >= 4],
            )
            b = merge_shards(root, shard_count=2, k=K, problem_ids=PROBLEMS)
            self.assertEqual(a, b)


class TestConflictingDuplicates(unittest.TestCase):
    def test_merge_raises_on_disagreement(self):
        a = _rec("pA", 0, verified=True)
        b = _rec("pA", 0, verified=False)
        with self.assertRaises(ShardConflictError):
            merge_records([a, b], k=1, problem_ids=["pA"])

    def test_identical_duplicates_kept_once(self):
        rec = _rec("pA", 0, verified=True)
        merged = merge_records([rec, dict(rec)], k=1, problem_ids=["pA"])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["verified"], True)

    def test_shard_pair_conflict_fails_merge(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_shard(root, ShardManifest(0, 2, 0, 1, None), [_rec("pA", 0, verified=True)])
            write_shard(root, ShardManifest(1, 2, 0, 1, None), [_rec("pA", 0, verified=False)])
            with self.assertRaises(ShardConflictError):
                merge_shards(root, shard_count=2, k=1, problem_ids=["pA"])


class TestResumeAfterPartialFailure(unittest.TestCase):
    def test_resume_skips_existing_keys_and_completes(self):
        records = _full_pool(k=4, problems=("pA",))
        manifest = ShardManifest(0, 1, 0, 4, None)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_shard(root, manifest, records[:2])
            keys_mid = existing_keys(root / "shards" / "000of001.jsonl")
            self.assertEqual(keys_mid, {("pA", 0), ("pA", 1)})
            # Crash mid-shard, then resume with the same manifest.
            write_shard(root, manifest, records, resume=True)
            keys_done = existing_keys(root / "shards" / "000of001.jsonl")
            self.assertEqual(keys_done, {("pA", i) for i in range(4)})
            payload = merge_shards(root, shard_count=1, k=4, problem_ids=("pA",))
            # Resume must not have written a second row for idx 0 or 1.
            raw = (root / "shards" / "000of001.jsonl").read_text(encoding="utf-8")
            self.assertEqual(raw.count('"sample_idx":0'), 1)
            self.assertEqual(raw.count('"sample_idx":1'), 1)
            self.assertEqual(payload, samples_jsonl_path(root).read_bytes())

    def test_resume_conflict_on_existing_key_raises(self):
        manifest = ShardManifest(0, 1, 0, 2, None)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_shard(root, manifest, [_rec("pA", 0, verified=True)])
            with self.assertRaises(ShardConflictError):
                write_shard(
                    root,
                    manifest,
                    [_rec("pA", 0, verified=False), _rec("pA", 1)],
                    resume=True,
                )

    def test_resume_rejects_manifest_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_shard(root, ShardManifest(0, 1, 0, 4, None), [_rec("pA", 0)])
            with self.assertRaises(ShardManifestError):
                write_shard(
                    root,
                    ShardManifest(0, 1, 0, 8, None),
                    [_rec("pA", 1)],
                    resume=True,
                )


class TestMergeCompletenessAndCanonicalBytes(unittest.TestCase):
    def test_gap_fails(self):
        recs = [_rec("pA", 0), _rec("pA", 2)]
        with self.assertRaises(MergeIncompleteError):
            merge_records(recs, k=3, problem_ids=["pA"])

    def test_empty_problem_ids_in_manifest_illegal(self):
        with self.assertRaises(ShardManifestError):
            ShardManifest(0, 1, 0, 1, ())

    def test_dumps_are_stable(self):
        rec = _rec("pA", 3, verified=True)
        self.assertEqual(
            dumps_record(rec),
            '{"problem_id":"pA","sample_idx":3,"response":"pA:3","verified":true,"n_tokens":13}',
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
