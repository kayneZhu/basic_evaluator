"""
Math adaptors that emit the Appendix C.1 prompt (not VerlPromptMixin).

Data loading and the Math-Verify cascade are the TeacherTraces / VerlAligned
schema. The user string is C.1. B0–B3 bins are NOT read from jsonl
``pass_count`` / ``pass_rate`` (those are the previous paper's pass@32).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from .base_adaptor import BaseAdaptor
from .c1_prompt_mixin import C1PromptMixin
from .verl_aligned_adaptor import _extract_boxed, _verify_math


def aime_union_problem_id(item: Dict[str, Any]) -> str:
    """
    ``data/aime24_25_26_bench_schema.jsonl`` concatenates the year files
    with ``unique_id`` 0–89. Year files reuse 0–29, so we namespace:

    0–29 → ``aime24:{uid}``; 30–59 → ``aime25:{uid-30}``; 60–89 → ``aime26:{uid-60}``.
    """
    uid = int(item["unique_id"])
    if 0 <= uid < 30:
        return f"aime24:{uid}"
    if 30 <= uid < 60:
        return f"aime25:{uid - 30}"
    if 60 <= uid < 90:
        return f"aime26:{uid - 60}"
    raise ValueError(
        f"aime_union unique_id {uid} is outside the 90-problem union [0, 90)"
    )


def or1_problem_id(item: Dict[str, Any]) -> str:
    return f"or1:{item['index']}"


def is_aime26_problem_id(problem_id: str) -> bool:
    """AIME26 is its own §5.1 row: filter the union pool, do not redraw."""
    return problem_id.startswith("aime26:")


class C1MathAdaptor(C1PromptMixin, BaseAdaptor):
    """question / ground_truth jsonl + C.1 prompt + Math-Verify cascade."""

    def _load_data(self) -> List[Dict[str, Any]]:
        data: List[Dict[str, Any]] = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    def get_question(self, item: Dict[str, Any]) -> str:
        return item.get("question", "")

    def get_ground_truth(self, item: Dict[str, Any]) -> str:
        return item.get("ground_truth", "")

    def extract_answer(self, model_output: str) -> str:
        return _extract_boxed(model_output)

    def verify_answer(self, model_answer: str, ground_truth: str) -> bool:
        return _verify_math(model_answer, ground_truth)

    def get_problem_id(self, item: Dict[str, Any]) -> str:
        raise NotImplementedError

    def get_variant_metadata(self, item: Dict[str, Any]) -> Dict[str, Any]:
        # Intentionally omit pass_count / pass_rate — those are not bins.
        return {"problem_id": self.get_problem_id(item)}


class C1AimeUnionAdaptor(C1MathAdaptor):
    def get_problem_id(self, item: Dict[str, Any]) -> str:
        return aime_union_problem_id(item)


class C1OR1200Adaptor(C1MathAdaptor):
    def get_problem_id(self, item: Dict[str, Any]) -> str:
        return or1_problem_id(item)
