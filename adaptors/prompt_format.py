"""C.1 / C.2 prompt selection for evaluation.

C.2 is a fixed ChatML string, the same bytes training renders in
``opd_frontier.data.prompt.render_c2_chatml``. Do not call a model's
``apply_chat_template`` for C.2: Qwen3-1.7B-Base must get the same ids
as Qwen3-1.7B and the Qwen3-4B teacher.

C.2 keeps C.1's system line and user text. The only difference from
C.1 is the assistant prefill: a closed empty think block instead of
an open ``<think>\\n``.
"""

from __future__ import annotations

import os
from typing import Any, Optional

PROMPT_FORMAT_C1 = "c1_think"
PROMPT_FORMAT_C2 = "c2_nothink"
PROMPT_FORMATS = (PROMPT_FORMAT_C1, PROMPT_FORMAT_C2)

C1_SYSTEM_PROMPT = (
    "You are an expert mathematician with strong problem-solving skills. "
    "Think step by step."
)
C2_USER_SUFFIX = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)
C2_ASSISTANT_PREFILL = "<think>\n\n</think>\n\n"


def c1_system_block() -> str:
    """Same ChatML system turn C.1's published template emits."""
    return f"<|im_start|>system\n{C1_SYSTEM_PROMPT}<|im_end|>\n"


def build_c2_user_content(problem: str) -> str:
    """Same characters as training ``build_c1_user_content``."""
    return f"{problem}\n{C2_USER_SUFFIX}"


def render_c2_chatml(user: str) -> str:
    """The one C.2 renderer. ``user`` is the full user-turn body.

    No BOS. Callers encode with ``add_special_tokens=False``.
    """
    return (
        f"{c1_system_block()}"
        "<|im_start|>user\n"
        f"{user}<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{C2_ASSISTANT_PREFILL}"
    )


def render_c2_prompt(problem: str) -> str:
    return render_c2_chatml(build_c2_user_content(problem))


def resolve_prompt_format(
    explicit: Optional[str] = None,
    *,
    required: Optional[bool] = None,
) -> str:
    """``c1_think`` or ``c2_nothink``.

    Unset stays ``c1_think`` unless ``required`` or
    ``OPD_PROMPT_FORMAT_REQUIRED=1``. Unknown values fail.
    """
    if required is None:
        flag = os.environ.get("OPD_PROMPT_FORMAT_REQUIRED", "")
        required = flag in {"1", "true", "True", "yes"}
    raw = explicit if explicit not in (None, "") else os.environ.get("OPD_PROMPT_FORMAT")
    if raw is None or str(raw).strip() == "":
        if required:
            raise RuntimeError(
                "OPD_PROMPT_FORMAT must be set to c1_think or c2_nothink"
            )
        return PROMPT_FORMAT_C1
    value = str(raw).strip()
    if value not in PROMPT_FORMATS:
        raise ValueError(
            f"OPD_PROMPT_FORMAT={value!r} is not one of {PROMPT_FORMATS}"
        )
    return value


def apply_cli_prompt_format(args: Any) -> str:
    """Resolve CLI / env and publish the choice for this process and workers."""
    explicit = getattr(args, "prompt_format", None)
    required = bool(getattr(args, "require_prompt_format", False))
    try:
        fmt = resolve_prompt_format(explicit, required=required)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    os.environ["OPD_PROMPT_FORMAT"] = fmt
    args.prompt_format = fmt
    return fmt


def prompt_format_label(fmt: Optional[str] = None) -> str:
    chosen = fmt or resolve_prompt_format()
    if chosen == PROMPT_FORMAT_C2:
        return "C.2 nothink <think>\\n\\n</think>\\n\\n"
    return "C.1 + <think>\\n"


def verifier_text(response: str, *, prompt_format: Optional[str] = None) -> str:
    """Whole completion. C.2 does not slice on ``</think>``."""
    resolve_prompt_format(prompt_format)
    return response or ""


def c2_response_think_flags(text: str) -> dict:
    """Literal think tags in the response. Diagnostics only.

    ``has_think_end`` stays separate: true when generated ids contain
    151668 (``</think>``) or the response text contains ``</think>``.
    Under C.2 the empty think block is in the prompt, so a hit in the
    continuation should be rare and does not gate the verifier.
    """
    body = text or ""
    return {
        "response_contains_think_open": "<think>" in body,
        "response_contains_think_close": "</think>" in body,
    }
