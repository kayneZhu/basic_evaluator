"""
STUB: HumanEval+ adaptor for the §5.1 zero-training coding benchmark.

This is not an implementation. HumanEval+ only has to answer 「有没有伤」,
but it still needs:

- the HumanEval+ / evalplus prompt + hidden tests
- extraction of a Python completion
- **sandboxed** execution (timeout, no network, no host FS writes)

A fake in-process ``exec`` would be worse than a missing adaptor. Every
entry point raises ``NotImplementedError`` so a paper job cannot silently
report zeros.

Lead still owes K for this surface (§5.1 does not state it; T=1 is frozen).
"""

from __future__ import annotations

from typing import Any, Dict, List

from .base_adaptor import BaseAdaptor


STUB_MESSAGE = (
    "STUB / NotImplementedError: HumanEval+ is the §5.1 zero-training coding "
    "benchmark (lead 2026-09-22) but the sandboxed runner is not implemented "
    "in this checkout. Do not invent pass@K from an unsandboxed exec. "
    "On the eval host: load evalplus HumanEval+, extract the completion, "
    "run hidden tests in a sandbox, write INTERFACE sample rows "
    "(problem_id, sample_idx, response, verified, n_tokens). K is still "
    "unspecified in §5.1."
)


class HumanEvalPlusAdaptor(BaseAdaptor):
    """Explicit stub. Construction itself fails so it cannot be scheduled."""

    def _load_data(self) -> List[Dict[str, Any]]:
        raise NotImplementedError(STUB_MESSAGE)

    def _get_system_prompt(self) -> str:
        raise NotImplementedError(STUB_MESSAGE)

    def format_prompt(self, item: Dict[str, Any]) -> str:
        raise NotImplementedError(STUB_MESSAGE)

    def extract_answer(self, model_output: str) -> str:
        raise NotImplementedError(STUB_MESSAGE)

    def verify_answer(self, model_answer: str, ground_truth: str) -> bool:
        raise NotImplementedError(STUB_MESSAGE)
