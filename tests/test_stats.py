#!/usr/bin/env python3
"""
Every §5.1 estimator against hand-computed values on a tiny fixture.

These numbers go in the paper. A wrong estimator is worse than a missing one.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opd_eval.samples import write_samples_jsonl
from opd_eval.stats import (
    DEFAULT_N_BOOT,
    StatsError,
    avg_at_8,
    delta_log_p_hat,
    learned_forgotten,
    load_pool,
    log_p_hat,
    mcnemar,
    mean_delta_log_p,
    mean_pass_at_k,
    pass_at_k_bit,
    percentile,
    per_problem_delta_log_p,
    per_problem_stats,
    pool_from_records,
    resample_verified,
    shrinkage_p_hat,
    two_level_bootstrap,
    two_level_bootstrap_paired,
    two_level_resample_paired,
    two_level_resample_pool,
    _binom_cdf_le,
    _chi2_sf_1df,
)


K = 8

# ckpt verified bits (sample_idx 0..7). Hand counts:
# p1 k=2 pass=1  p2 k=0 pass=0  p3 k=6 pass=1
CKPT = {
    "p1": (True, False, False, False, True, False, False, False),
    "p2": (False, False, False, False, False, False, False, False),
    "p3": (True, True, True, True, True, True, False, False),
}

# base: p1 unsolvable; p2 and p3 solvable; p2 will be forgotten, p1 learned
BASE = {
    "p1": (False, False, False, False, False, False, False, False),
    "p2": (True, False, False, False, False, False, False, False),
    "p3": (True, True, True, True, True, True, True, True),
}

BINS = {"p1": "B0", "p2": "B1", "p3": "B3"}


def _records(pool, k=K):
    recs = []
    for pid, bits in pool.items():
        for idx, ok in enumerate(bits):
            recs.append({
                "problem_id": pid,
                "sample_idx": idx,
                "response": f"{pid}:{idx}",
                "verified": bool(ok),
                "n_tokens": 1,
            })
    return recs


class ScriptedRNG:
    def __init__(self, draws):
        self.draws = list(draws)

    def randrange(self, n):
        value = self.draws.pop(0)
        if not (0 <= value < n):
            raise AssertionError(f"scripted draw {value} not in [0, {n})")
        return value


class TestHandComputedEstimators(unittest.TestCase):
    def test_shrinkage_and_log(self):
        self.assertEqual(shrinkage_p_hat(0, 8), 0.5 / 9)
        self.assertEqual(shrinkage_p_hat(2, 8), 2.5 / 9)
        self.assertEqual(shrinkage_p_hat(6, 8), 6.5 / 9)
        self.assertEqual(shrinkage_p_hat(8, 8), 8.5 / 9)
        self.assertAlmostEqual(log_p_hat(2, 8), math.log(2.5 / 9))
        self.assertAlmostEqual(delta_log_p_hat(0, 2, 8), math.log(5.0))
        self.assertAlmostEqual(delta_log_p_hat(1, 0, 8), math.log((0.5 / 9) / (1.5 / 9)))
        self.assertEqual(pass_at_k_bit(0), 0)
        self.assertEqual(pass_at_k_bit(1), 1)
        self.assertEqual(pass_at_k_bit(8), 1)

    def test_pass_at_k_and_avg8(self):
        self.assertAlmostEqual(mean_pass_at_k(CKPT, K), 2.0 / 3.0)
        # avg@8 = mean of verified on idx 0–7: (2 + 0 + 6) / 24 = 1/3
        self.assertAlmostEqual(avg_at_8(CKPT), 1.0 / 3.0)
        stats = per_problem_stats(CKPT, K)
        self.assertEqual(stats["p1"].k, 2)
        self.assertEqual(stats["p1"].pass_bit, 1)
        self.assertEqual(stats["p2"].k, 0)
        self.assertEqual(stats["p2"].pass_bit, 0)
        self.assertEqual(stats["p3"].k, 6)
        self.assertAlmostEqual(stats["p1"].avg8, 2.0 / 8.0)
        self.assertAlmostEqual(stats["p2"].avg8, 0.0)
        self.assertAlmostEqual(stats["p3"].avg8, 6.0 / 8.0)
        self.assertAlmostEqual(stats["p1"].p_hat, 2.5 / 9)
        self.assertAlmostEqual(stats["p2"].p_hat, 0.5 / 9)
        self.assertAlmostEqual(stats["p3"].p_hat, 6.5 / 9)

    def test_avg8_refuses_short_pool(self):
        short = {"p": (True, False, True, False)}
        with self.assertRaises(StatsError):
            avg_at_8(short)

    def test_delta_log_p_hat_per_problem(self):
        deltas = per_problem_delta_log_p(BASE, CKPT, K)
        self.assertAlmostEqual(deltas["p1"], math.log(5.0))
        self.assertAlmostEqual(deltas["p2"], math.log(1.0 / 3.0))
        self.assertAlmostEqual(deltas["p3"], math.log(6.5 / 8.5))
        expected_mean = (math.log(5.0) + math.log(1.0 / 3.0) + math.log(6.5 / 8.5)) / 3.0
        self.assertAlmostEqual(mean_delta_log_p(BASE, CKPT), expected_mean)
        self.assertAlmostEqual(expected_mean, math.log(65.0 / 51.0) / 3.0)

    def test_learned_forgotten_and_or1_denoms(self):
        lf = learned_forgotten(BASE, CKPT, K, bins=BINS)
        self.assertEqual(lf["learned"], 1)
        self.assertEqual(lf["forgotten"], 1)
        self.assertEqual(lf["net"], 0)
        self.assertEqual(lf["n_base_unsolvable"], 1)
        self.assertEqual(lf["n_base_solvable"], 2)
        self.assertAlmostEqual(lf["learned_rate"], 1.0)
        self.assertAlmostEqual(lf["forgotten_rate"], 0.5)

        self.assertEqual(lf["by_bin"]["B0"]["learned"], 1)
        self.assertEqual(lf["by_bin"]["B0"]["forgotten"], 0)
        self.assertEqual(lf["by_bin"]["B0"]["n_base_unsolvable"], 1)
        self.assertEqual(lf["by_bin"]["B0"]["n_base_solvable"], 0)
        self.assertAlmostEqual(lf["by_bin"]["B0"]["learned_rate"], 1.0)
        self.assertIsNone(lf["by_bin"]["B0"]["forgotten_rate"])

        self.assertEqual(lf["by_bin"]["B1"]["learned"], 0)
        self.assertEqual(lf["by_bin"]["B1"]["forgotten"], 1)
        self.assertIsNone(lf["by_bin"]["B1"]["learned_rate"])
        self.assertAlmostEqual(lf["by_bin"]["B1"]["forgotten_rate"], 1.0)

        self.assertEqual(lf["by_bin"]["B3"]["learned"], 0)
        self.assertEqual(lf["by_bin"]["B3"]["forgotten"], 0)
        self.assertAlmostEqual(lf["by_bin"]["B3"]["forgotten_rate"], 0.0)

    def test_mcnemar_hand(self):
        result = mcnemar(BASE, CKPT, K)
        self.assertEqual(result.learned, 1)
        self.assertEqual(result.forgotten, 1)
        self.assertEqual(result.n_discordant, 2)
        self.assertEqual(result.chi2, 0.0)
        self.assertEqual(result.p_chi2, 1.0)
        self.assertEqual(result.p_exact, 1.0)

        # b=3, c=0 constructed as three learned problems, K=1
        base = {f"q{i}": (False,) for i in range(3)}
        ckpt = {f"q{i}": (True,) for i in range(3)}
        r = mcnemar(base, ckpt, 1)
        self.assertEqual(r.learned, 3)
        self.assertEqual(r.forgotten, 0)
        self.assertAlmostEqual(r.chi2, 3.0)
        self.assertAlmostEqual(r.p_chi2, _chi2_sf_1df(3.0))
        self.assertAlmostEqual(r.p_exact, 0.25)
        self.assertAlmostEqual(_binom_cdf_le(3, 0), 0.125)

    def test_chi2_sf_matches_erfc(self):
        self.assertEqual(_chi2_sf_1df(0.0), 1.0)
        self.assertAlmostEqual(_chi2_sf_1df(3.0), math.erfc(math.sqrt(1.5)))


class TestTwoLevelBootstrap(unittest.TestCase):
    def test_resample_verified_uses_global_indices(self):
        bits = (True, False, False)
        got = resample_verified(bits, ScriptedRNG([2, 0, 2]))
        self.assertEqual(got, (False, True, False))

    def test_two_level_replicate_hand(self):
        pool = {"p": (True, False)}
        # draw the only problem, then both completions from idx 0 → (T, T)
        star = two_level_resample_pool(pool, ScriptedRNG([0, 0, 0]))
        self.assertEqual(list(star.values()), [(True, True)])
        k = sum(star[next(iter(star))])
        self.assertEqual(k, 2)
        self.assertAlmostEqual(shrinkage_p_hat(k, 2), 2.5 / 3.0)
        self.assertEqual(pass_at_k_bit(k), 1)

        # both from idx 1 → (F, F): pass@2 = 0, p̂ = 0.5/3
        star0 = two_level_resample_pool(pool, ScriptedRNG([0, 1, 1]))
        k0 = sum(star0[next(iter(star0))])
        self.assertEqual(k0, 0)
        self.assertEqual(pass_at_k_bit(k0), 0)
        self.assertAlmostEqual(shrinkage_p_hat(k0, 2), 0.5 / 3.0)

    def test_paired_resample_independent_completions(self):
        base = {"p": (True, False)}
        ckpt = {"p": (False, True)}
        # problem draw 0; base completions 0,0 → (T,T); ckpt 1,1 → (T,T)
        b_star, c_star = two_level_resample_paired(
            base, ckpt, ScriptedRNG([0, 0, 0, 1, 1])
        )
        self.assertEqual(list(b_star.values()), [(True, True)])
        self.assertEqual(list(c_star.values()), [(True, True)])
        key = next(iter(b_star))
        self.assertAlmostEqual(
            delta_log_p_hat(2, 2, 2),
            log_p_hat(2, 2) - log_p_hat(2, 2),
        )
        self.assertEqual(key, next(iter(c_star)))

    def test_percentile_linear(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertEqual(percentile(xs, 0.0), 1.0)
        self.assertEqual(percentile(xs, 1.0), 5.0)
        self.assertEqual(percentile(xs, 0.5), 3.0)
        self.assertEqual(percentile(xs, 0.25), 2.0)

    def test_default_n_boot_is_10000(self):
        self.assertEqual(DEFAULT_N_BOOT, 10000)
        self.assertEqual(two_level_bootstrap.__kwdefaults__["n_boot"], 10000)
        self.assertEqual(
            two_level_bootstrap_paired.__kwdefaults__["n_boot"], 10000
        )


class TestPoolFromJsonl(unittest.TestCase):
    def test_roundtrip_and_gap_fails(self):
        recs = _records(CKPT)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "samples.jsonl"
            write_samples_jsonl(path, recs)
            pool = load_pool(path, K)
            self.assertEqual(pool["p1"], CKPT["p1"])
            self.assertAlmostEqual(avg_at_8(pool), 1.0 / 3.0)

        gapped = [r for r in recs if not (r["problem_id"] == "p1" and r["sample_idx"] == 3)]
        with self.assertRaises(StatsError):
            pool_from_records(gapped, K)


if __name__ == "__main__":
    unittest.main(verbosity=2)
