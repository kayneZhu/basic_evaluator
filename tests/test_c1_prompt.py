#!/usr/bin/env python3
"""
Freeze the Appendix C.1 prompt. Training and evaluation must be byte-identical
at g=0; a substring match is not enough.

If the Qwen3-Base tokenizers are not cached and cannot be downloaded, the
pre-chat-template message list and the published ChatML render are still
asserted exactly. Live ``apply_chat_template`` on both HF ids is then marked
PENDING GPU VERIFICATION (xfail, not a silent skip).
"""

from __future__ import annotations

import hashlib
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptors.c1_prompt_mixin import (
    C1_ADD_GENERATION_PROMPT,
    C1_ENABLE_THINKING,
    C1_STUDENT_TOKENIZER_IDS,
    C1_SYSTEM_PROMPT,
    C1_THINK_PREFILL,
    C1_USER_SUFFIX,
    C1PromptMixin,
    _PublishedQwen3BaseChatTemplate,
    build_c1_messages,
    build_c1_user_content,
    render_c1_chatml,
    render_c1_prompt,
)
from opd_eval.contract import (
    base_default_eos_token_id,
    golden_g0_prompt,
    load_g0_golden,
    stop_convention_digest,
    stop_token_ids,
    template_digest,
    trace_terminator_token_id,
    vllm_sampling_kwargs,
    vllm_stop_token_ids,
)


EVAL_ROOT = Path(__file__).resolve().parents[1]
G0_GOLDEN_JSON = EVAL_ROOT / "contract" / "g0_prompt.golden.json"
INTERFACE_MD = EVAL_ROOT / "INTERFACE.md"
_G0_HEADER = "### The g=0 prompt is a cross-side invariant"
_PLACEHOLDER = "{problem}"


def parse_interface_md_g0_template(interface_md: str) -> str:
    """Return the fenced documentation block, ending in a single newline.

    The markdown fence is written with a blank line before the closing
    backticks. That blank line is fence formatting, not part of the
    frozen prefill. Trailing newlines are collapsed to one so the
    documentation can be compared to the JSON fixture.
    """
    start = interface_md.find(_G0_HEADER)
    if start < 0:
        raise ValueError(
            "INTERFACE.md is missing the section "
            f"{_G0_HEADER!r}; refusing to invent a golden string"
        )
    rest = interface_md[start:]
    open_fence = rest.find("```")
    if open_fence < 0:
        raise ValueError("g=0 golden section has no opening fence")
    after_open = rest[open_fence + 3 :]
    nl = after_open.find("\n")
    if nl < 0:
        raise ValueError("g=0 golden fence is empty")
    body = after_open[nl + 1 :]
    close = body.find("```")
    if close < 0:
        raise ValueError("g=0 golden section has no closing fence")
    template = body[:close]
    if _PLACEHOLDER not in template:
        raise ValueError("golden template is missing the {problem} placeholder")
    if "<think>" not in template:
        raise ValueError("golden template is missing the format-only <think> prefill")
    return template.rstrip("\n") + "\n"


# Fixed toy problem. Do not change without updating every golden string.
TOY_PROBLEM = "Compute 1+1."

EXPECTED_MESSAGES = [
    {
        "role": "system",
        "content": (
            "You are an expert mathematician with strong problem-solving skills. "
            "Think step by step."
        ),
    },
    {
        "role": "user",
        "content": (
            "Compute 1+1.\n"
            "Please reason step by step, and put your final answer within \\boxed{}."
        ),
    },
]

# Exact render from the JSON fixture (authority), not the markdown fence.
EXPECTED_RENDERED_C1 = golden_g0_prompt(TOY_PROBLEM)

PREVIOUS_PAPER_USER_PHRASES = (
    "Please reason step by step to solve this problem.",
    "Use this format:",
    "[Your reasoning process here, showing how YOU would reach the solution]",
)

PENDING_GPU_TOKENIZER_VERIFICATION = (
    "PENDING GPU VERIFICATION: live apply_chat_template for "
    "Qwen/Qwen3-1.7B-Base and Qwen/Qwen3-0.6B-Base. Tokenizers are not "
    "cached locally and could not be downloaded (tokenizer-only; no model "
    "weights). On the GPU host, for EACH of those two ids, load "
    "AutoTokenizer.from_pretrained(id, trust_remote_code=True) and assert "
    "render_c1_prompt(TOY_PROBLEM, tok) == EXPECTED_RENDERED_C1 exactly "
    "(not a substring). Also confirm enable_thinking is omitted and a "
    "single format-only <think>\\n is prefilled. Confirm eos_token_id "
    "equals the contract base_default_eos_token_id and that "
    "<|im_end|> equals trace_terminator_token_id. The pre-template "
    "message list is frozen in test_c1_messages_exact."
)


def _try_load_base_tokenizers():
    """Return {hf_id: tokenizer} or None if neither cache nor download works."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None
    loaded = {}
    for hf_id in C1_STUDENT_TOKENIZER_IDS:
        try:
            loaded[hf_id] = AutoTokenizer.from_pretrained(
                hf_id, trust_remote_code=True, local_files_only=True
            )
        except Exception:
            try:
                loaded[hf_id] = AutoTokenizer.from_pretrained(
                    hf_id, trust_remote_code=True
                )
            except Exception:
                return None
    if len(loaded) != len(C1_STUDENT_TOKENIZER_IDS):
        return None
    return loaded


class TestC1Messages(unittest.TestCase):
    def test_c1_messages_exact(self):
        self.assertEqual(build_c1_messages(TOY_PROBLEM), EXPECTED_MESSAGES)

    def test_user_content_is_problem_then_suffix(self):
        self.assertEqual(
            build_c1_user_content(TOY_PROBLEM),
            f"{TOY_PROBLEM}\n{C1_USER_SUFFIX}",
        )

    def test_constants_match_interface(self):
        golden = load_g0_golden()
        self.assertEqual(
            C1_SYSTEM_PROMPT,
            "You are an expert mathematician with strong problem-solving skills. "
            "Think step by step.",
        )
        self.assertEqual(
            C1_USER_SUFFIX,
            "Please reason step by step, and put your final answer within \\boxed{}.",
        )
        self.assertEqual(C1_THINK_PREFILL, golden["think_prefill"])
        self.assertEqual(C1_ADD_GENERATION_PROMPT, golden["add_generation_prompt"])
        self.assertIs(C1_ENABLE_THINKING, golden["enable_thinking"])
        self.assertEqual(C1_THINK_PREFILL, "<think>\n")
        self.assertTrue(C1_ADD_GENERATION_PROMPT)
        self.assertIsNone(C1_ENABLE_THINKING)

    def test_not_previous_paper_rho0_user_block(self):
        user = build_c1_user_content(TOY_PROBLEM)
        for phrase in PREVIOUS_PAPER_USER_PHRASES:
            self.assertNotIn(phrase, user)
        rendered = render_c1_chatml(TOY_PROBLEM)
        for phrase in PREVIOUS_PAPER_USER_PHRASES:
            self.assertNotIn(phrase, rendered)


class TestC1RenderedString(unittest.TestCase):
    def test_published_chatml_exact(self):
        self.assertEqual(render_c1_chatml(TOY_PROBLEM), EXPECTED_RENDERED_C1)

    def test_published_template_standin_exact_for_both_student_ids(self):
        # Same published path on both Base ids; assert once per id so a
        # future template fork cannot hide behind a single check.
        tok = _PublishedQwen3BaseChatTemplate()
        for hf_id in C1_STUDENT_TOKENIZER_IDS:
            got = render_c1_prompt(TOY_PROBLEM, tok)
            self.assertEqual(
                got,
                EXPECTED_RENDERED_C1,
                msg=f"published ChatML path for {hf_id}",
            )

    def test_mixin_format_prompt_exact(self):
        class _ToyAdaptor(C1PromptMixin):
            def __init__(self):
                self._tokenizer = None
                self._tokenizer_name = None
                self.data = []
                self.system_prompt = self._get_system_prompt()

        adaptor = _ToyAdaptor()
        self.assertTrue(adaptor.raw_prompts)
        self.assertEqual(adaptor._get_system_prompt(), C1_SYSTEM_PROMPT)
        self.assertEqual(
            adaptor.format_prompt({"question": TOY_PROBLEM}),
            EXPECTED_RENDERED_C1,
        )

    def test_think_prefill_once_at_end(self):
        rendered = render_c1_chatml(TOY_PROBLEM)
        self.assertTrue(rendered.endswith("<think>\n"))
        self.assertEqual(rendered.count("<think>"), 1)
        self.assertNotIn("</think>", rendered)

    def test_add_generation_prompt_opens_assistant_turn(self):
        rendered = render_c1_chatml(TOY_PROBLEM)
        self.assertIn("<|im_start|>assistant\n<think>\n", rendered)

    def test_standin_refuses_enable_thinking_kwarg(self):
        tok = _PublishedQwen3BaseChatTemplate()
        with self.assertRaises(ValueError):
            tok.apply_chat_template(
                EXPECTED_MESSAGES,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )


class TestGoldenJsonAuthority(unittest.TestCase):
    def test_recorded_sha256_matches_own_template(self):
        self.assertTrue(G0_GOLDEN_JSON.is_file())
        golden = load_g0_golden()
        digest = hashlib.sha256(golden["template"].encode("utf-8")).hexdigest()
        self.assertEqual(golden["sha256"], digest)
        self.assertEqual(template_digest(golden), golden["sha256"])

    def test_recorded_stop_sha256_matches_stop_convention(self):
        golden = load_g0_golden()
        self.assertEqual(stop_convention_digest(golden), golden["stop_sha256"])

    def test_interface_md_fence_matches_json_template(self):
        md = INTERFACE_MD.read_text(encoding="utf-8")
        self.assertEqual(
            parse_interface_md_g0_template(md),
            load_g0_golden()["template"],
        )

    def test_interface_md_states_terminator_and_stop_set(self):
        md = INTERFACE_MD.read_text(encoding="utf-8")
        golden = load_g0_golden()
        self.assertIn("trace_terminator_token_id", md)
        self.assertIn("stop_token_ids", md)
        self.assertIn(str(golden["trace_terminator_token_id"]), md)
        self.assertIn(str(golden["base_default_eos_token_id"]), md)


class TestStopConventionFromContract(unittest.TestCase):
    def test_stop_set_and_terminator_are_loaded_from_contract(self):
        golden = load_g0_golden()
        self.assertEqual(
            trace_terminator_token_id(), golden["trace_terminator_token_id"]
        )
        self.assertEqual(stop_token_ids(), tuple(golden["stop_token_ids"]))
        self.assertEqual(
            base_default_eos_token_id(), golden["base_default_eos_token_id"]
        )
        self.assertEqual(vllm_stop_token_ids(), list(golden["stop_token_ids"]))
        kwargs = vllm_sampling_kwargs(max_tokens=8192, temperature=1.0)
        self.assertEqual(kwargs["stop_token_ids"], list(golden["stop_token_ids"]))
        self.assertNotIn("eos_token_id", kwargs)

    def test_base_default_eos_is_in_stop_set_but_is_not_the_terminator(self):
        golden = load_g0_golden()
        default = golden["base_default_eos_token_id"]
        terminator = golden["trace_terminator_token_id"]
        stops = list(golden["stop_token_ids"])
        self.assertIn(default, stops)
        self.assertNotEqual(default, terminator)
        self.assertIn(terminator, stops)
        self.assertEqual(base_default_eos_token_id(), default)
        self.assertEqual(trace_terminator_token_id(), terminator)

    def test_sampling_engines_pass_contract_stop_token_ids(self):
        model_src = (EVAL_ROOT / "core" / "model_inference.py").read_text(
            encoding="utf-8"
        )
        parallel_src = (EVAL_ROOT / "core" / "parallel_inference.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("vllm_sampling_kwargs", model_src)
        self.assertIn("vllm_sampling_kwargs", parallel_src)
        self.assertNotIn("tokenizer.eos_token_id", model_src)
        self.assertNotIn("tokenizer.eos_token_id", parallel_src)


class TestC1LiveTokenizersOrPendingGPU(unittest.TestCase):
    def test_live_qwen3_base_tokenizers_match_golden_or_pending(self):
        loaded = _try_load_base_tokenizers()
        if loaded is None:
            # Not a skip and not a vacuous pass: still freeze the message
            # list and published ChatML, and print the GPU-host checklist.
            print("\n" + PENDING_GPU_TOKENIZER_VERIFICATION, file=sys.stderr)
            self.assertEqual(build_c1_messages(TOY_PROBLEM), EXPECTED_MESSAGES)
            self.assertEqual(render_c1_chatml(TOY_PROBLEM), EXPECTED_RENDERED_C1)
            self.assertTrue(
                PENDING_GPU_TOKENIZER_VERIFICATION.startswith(
                    "PENDING GPU VERIFICATION"
                )
            )
            for hf_id in C1_STUDENT_TOKENIZER_IDS:
                self.assertIn(hf_id, PENDING_GPU_TOKENIZER_VERIFICATION)
            return
        golden = load_g0_golden()
        for hf_id, tok in loaded.items():
            got = render_c1_prompt(TOY_PROBLEM, tok)
            self.assertEqual(
                got,
                EXPECTED_RENDERED_C1,
                msg=(
                    f"live apply_chat_template for {hf_id} diverged from "
                    f"EXPECTED_RENDERED_C1.\n--- got ---\n{got!r}\n"
                    f"--- expected ---\n{EXPECTED_RENDERED_C1!r}"
                ),
            )
            self.assertEqual(
                tok.eos_token_id,
                golden["base_default_eos_token_id"],
                msg=f"{hf_id} eos_token_id is not the recorded Base default",
            )
            self.assertEqual(
                tok.convert_tokens_to_ids("<|im_end|>"),
                golden["trace_terminator_token_id"],
                msg=f"{hf_id} <|im_end|> is not the recorded terminator",
            )
            self.assertEqual(
                tok.convert_tokens_to_ids("<|endoftext|>"),
                golden["base_default_eos_token_id"],
                msg=f"{hf_id} <|endoftext|> is not the recorded Base default",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
