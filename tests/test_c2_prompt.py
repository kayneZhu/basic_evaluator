"""C.2 prompt string, verifier, and fail-fast selection."""

from __future__ import annotations

import os
import unittest

from adaptors.c1_prompt_mixin import C1_SYSTEM_PROMPT as MIXIN_SYSTEM
from adaptors.c1_prompt_mixin import render_c1_chatml
from adaptors.prompt_format import (
    C1_SYSTEM_PROMPT,
    C2_ASSISTANT_PREFILL,
    build_c2_user_content,
    c1_system_block,
    render_c2_prompt,
    resolve_prompt_format,
    verifier_text,
)
from adaptors.verl_aligned_adaptor import _extract_boxed, _verify_math
from opd_eval.mini_protocol import completion_diag, main

PROBLEM = "Compute 1+1."
OLD_AND_NEW_USER = (
    "Compute 1+1.\nPlease reason step by step, and put your final answer within \\boxed{}."
)


class C2PromptTests(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("OPD_PROMPT_FORMAT", None)
        os.environ.pop("OPD_PROMPT_FORMAT_REQUIRED", None)

    def test_user_text_and_fixed_string(self):
        self.assertEqual(build_c2_user_content(PROBLEM), OLD_AND_NEW_USER)
        self.assertNotIn("<think>", OLD_AND_NEW_USER)
        rendered = render_c2_prompt(PROBLEM)
        self.assertEqual(C1_SYSTEM_PROMPT, MIXIN_SYSTEM)
        self.assertTrue(rendered.startswith(c1_system_block()))
        self.assertIn("<|im_start|>user\n" + OLD_AND_NEW_USER, rendered)
        self.assertTrue(rendered.endswith(C2_ASSISTANT_PREFILL))
        self.assertEqual(rendered.count("<|im_start|>system"), 1)
        c1 = render_c1_chatml(PROBLEM)
        self.assertTrue(c1.startswith(c1_system_block()))
        self.assertIn("Think step by step.", c1)
        self.assertNotIn("</think>", c1)
        self.assertTrue(c1.endswith("<think>\n"))

    def test_unset_defaults_c1_required_fails(self):
        self.assertEqual(resolve_prompt_format(), "c1_think")
        with self.assertRaises(RuntimeError):
            resolve_prompt_format(required=True)

    def test_verifier_whole_response_with_and_without_tags(self):
        plain = r"\boxed{27}"
        tagged = "<think>\nwork\n</think>\n\\boxed{27}"
        before = "\\boxed{27}</think>"
        for text in (plain, tagged, before):
            scored = verifier_text(text, prompt_format="c2_nothink")
            self.assertEqual(scored, text)
            self.assertEqual(_extract_boxed(scored), "27")
            self.assertTrue(_verify_math("27", "27"))
        self.assertFalse(_verify_math("", "27"))

    def test_c2_diag_records_think_strings_without_gating(self):
        os.environ["OPD_PROMPT_FORMAT"] = "c2_nothink"
        plain = completion_diag(r"\boxed{27}", [1, 2, 151645], "stop", max_new_tokens=32)
        self.assertFalse(plain["response_contains_think_open"])
        self.assertFalse(plain["response_contains_think_close"])
        self.assertEqual(plain["terminal_token_id"], 151645)
        self.assertFalse(plain["truncated"])
        self.assertFalse(plain["has_think_end"])
        tagged = completion_diag(
            "</think>\n\\boxed{27}",
            [151668, 151645],
            "stop",
            max_new_tokens=32,
        )
        self.assertTrue(tagged["response_contains_think_close"])
        self.assertTrue(tagged["has_think_end"])
        self.assertEqual(tagged["terminal_token_id"], 151645)

    def test_c1_diag_omits_c2_string_flags(self):
        diag = completion_diag(r"\boxed{27}", [1, 151645], "stop")
        self.assertNotIn("response_contains_think_open", diag)
        self.assertEqual(diag["terminal_token_id"], 151645)

    def test_cli_require_fails_when_unset(self):
        with self.assertRaises(SystemExit):
            main(
                [
                    "--out-root",
                    "/tmp/opd-c2-unset",
                    "--require-prompt-format",
                    "--dry-plan",
                ]
            )


if __name__ == "__main__":
    unittest.main()
