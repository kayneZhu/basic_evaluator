"""C.2 G1 table and the 50-gram repetition flag."""

from __future__ import annotations

import os
import unittest

from opd_eval.g1_hmini import g1_table, prompt_id_report
from opd_eval.mini_protocol import completion_diag, repeated_ngram


class RepetitionTests(unittest.TestCase):
    def test_short_and_unique_are_not_repetitions(self):
        self.assertFalse(repeated_ngram([1] * 49))
        self.assertFalse(repeated_ngram(list(range(80))))

    def test_same_50gram_three_times(self):
        gram = list(range(50))
        ids = gram + [99] + gram + [98] + gram
        self.assertTrue(repeated_ngram(ids))
        self.assertFalse(repeated_ngram(gram + gram))

    def test_diag_records_the_flag(self):
        os.environ["OPD_PROMPT_FORMAT"] = "c2_nothink"
        try:
            plain = completion_diag("boxed", [1, 2, 3], "stop", max_new_tokens=32)
            self.assertFalse(plain["repetition"])
            gram = list(range(50))
            looped = gram + gram + gram
            flagged = completion_diag("x", looped, "stop", max_new_tokens=10000)
            self.assertTrue(flagged["repetition"])
        finally:
            os.environ.pop("OPD_PROMPT_FORMAT", None)


class TableTests(unittest.TestCase):
    def test_per_bin_pass_and_rates(self):
        samples = []
        bin_of = {"or1:a": "B0", "or1:b": "B0"}
        # n=2. a is 2/2 correct, b is 0/2. One truncated, one repetition, one think tag.
        specs = {
            "or1:a": [(True, False, False, False), (True, True, False, True)],
            "or1:b": [(False, False, True, False), (False, False, False, False)],
        }
        for pid, rows in specs.items():
            for idx, (ok, trunc, rep, think) in enumerate(rows):
                samples.append(
                    {
                        "problem_id": pid,
                        "sample_idx": idx,
                        "verified": ok,
                        "n_tokens": 10 + idx,
                        "truncated": trunc,
                        "repetition": rep,
                        "terminal_token_id": 151645,
                        "response_contains_think_open": think,
                        "response_contains_think_close": False,
                    }
                )
        table = g1_table(samples, bin_of, n=2)
        bin0 = table["per_bin"]["B0"]
        self.assertEqual(bin0["n_problems"], 2)
        self.assertAlmostEqual(bin0["pass@1"], 0.5)
        self.assertAlmostEqual(bin0["pass@2"], 0.5)
        self.assertAlmostEqual(bin0["trunc_rate"], 0.25)
        self.assertAlmostEqual(bin0["repetition_rate"], 0.25)
        self.assertAlmostEqual(bin0["think_tag_rate"], 0.25)
        self.assertEqual(bin0["terminal_ids"], {"151645": 4})

    def test_prompt_ids_follow_the_strings(self):
        def render(problem: str) -> str:
            return f"P:{problem}"

        def encode(text: str):
            return [ord(ch) for ch in text]

        report = prompt_id_report(
            ["a", "b"],
            train_render=render,
            eval_render=render,
            encode=encode,
        )
        self.assertTrue(report["equal"])
        self.assertEqual(report["n_id_mismatch"], 0)
        bad = prompt_id_report(
            ["a"],
            train_render=render,
            eval_render=lambda problem: "other",
            encode=encode,
        )
        self.assertFalse(bad["equal"])
        self.assertEqual(bad["n_id_mismatch"], 1)


if __name__ == "__main__":
    unittest.main()
