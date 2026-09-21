"""
Appendix C.1 student prompt mixin for this paper's evaluation.

Do NOT reuse ``VerlPromptMixin``. That mixin pairs the correct C.1 *system*
message with the previous paper's ρ=0 user block
(``Please reason step by step to solve this problem.`` plus a
``Use this format:\\n<think>…`` scaffold). Config cannot fix it. A prompt
divergence between training at g=0 and evaluation silently invalidates
every comparison in the paper.

Contract (INTERFACE.md + proposal Appendix C.1 / C.3):

- System: ``You are an expert mathematician with strong problem-solving
  skills. Think step by step.``
- User: the problem text, then a newline, then
  ``Please reason step by step, and put your final answer within \\boxed{}.``
- Evaluation is always g=0: chat template only; no teacher prefix or hint.

Chat-template decisions (must stay byte-identical with training at g=0):

- ``add_generation_prompt=True``. The official Qwen3-Base template then
  emits ``<|im_start|>assistant\\n`` so generation starts on the assistant
  turn. ``False`` would leave the model continuing the user message.
- ``enable_thinking`` is **not** passed. The same official template, when
  ``enable_thinking is false``, injects a *closed* empty think block
  (``<think>\\n\\n</think>\\n\\n``) that skips thinking — the opposite of
  C.3's format-only open ``<think>\\n``. Passing ``True`` is equivalent
  to leaving it undefined on this path; we omit the kwarg so a TypeError
  cannot hide a template fork.
- A formatting ``<think>\\n`` **is** prefilled on the assistant side.
  Appendix C.3's g=0 row allows exactly this (「可含格式性 <think>\\n」,
  no teacher text). Teacher traces open with ``<think>``; Base students
  need the format cue; ``VerlPromptMixin`` already appends it after the
  template (we keep that *prefill*, not its ρ=0 user block).

``raw_prompts=True`` so the vLLM engine does not apply a second chat
template on top of this string.

The published ``chat_template`` on both ``Qwen/Qwen3-1.7B-Base`` and
``Qwen/Qwen3-0.6B-Base`` (Hugging Face ``tokenizer_config.json``, fetched
2026-09-22) is the same for the no-tools / system+user /
``add_generation_prompt=True`` / ``enable_thinking``-undefined path used
here. ``render_c1_chatml`` is that path plus the format prefill. Live
``apply_chat_template`` on the GPU host must still be asserted against
the same golden string (see ``tests/test_c1_prompt.py``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


C1_SYSTEM_PROMPT = (
    "You are an expert mathematician with strong problem-solving skills. "
    "Think step by step."
)

C1_USER_SUFFIX = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

# Appendix C.3 g=0 format-only assistant prefill. Not teacher content.
C1_THINK_PREFILL = "<think>\n"

# Frozen chat-template flags. Do not "fix" these without a training-side match.
C1_ADD_GENERATION_PROMPT = True
# Sentinel: we deliberately do not pass enable_thinking to apply_chat_template.
C1_ENABLE_THINKING: Optional[bool] = None

# Hugging Face ids whose chat templates this mixin must match.
C1_STUDENT_TOKENIZER_IDS: Sequence[str] = (
    "Qwen/Qwen3-1.7B-Base",
    "Qwen/Qwen3-0.6B-Base",
)


def build_c1_user_content(problem: str) -> str:
    """User-side C.1 string: problem text, then the boxed-answer instruction."""
    return f"{problem}\n{C1_USER_SUFFIX}"


def build_c1_messages(problem: str) -> List[Dict[str, str]]:
    """Pre-chat-template message list. The CPU-test freeze when no tokenizer."""
    return [
        {"role": "system", "content": C1_SYSTEM_PROMPT},
        {"role": "user", "content": build_c1_user_content(problem)},
    ]


def render_c1_chatml(problem: str) -> str:
    """
    Exact render of the official Qwen3-Base chat template on the g=0 path,
    plus the C.3 format-only ``<think>\\n`` prefill.

    This is the no-tools branch of the published Base ``chat_template``:

        <|im_start|>system\\n{system}<|im_end|>\\n
        <|im_start|>user\\n{user}<|im_end|>\\n
        <|im_start|>assistant\\n
        <think>\\n
    """
    messages = build_c1_messages(problem)
    parts = []
    for message in messages:
        parts.append(
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
        )
    if C1_ADD_GENERATION_PROMPT:
        parts.append("<|im_start|>assistant\n")
    parts.append(C1_THINK_PREFILL)
    return "".join(parts)


def render_c1_prompt(problem: str, tokenizer: Any) -> str:
    """
    Render with a real tokenizer's ``apply_chat_template``.

    Flags: ``tokenize=False``, ``add_generation_prompt=True``. Does **not**
    pass ``enable_thinking``. Then appends ``C1_THINK_PREFILL``.
    """
    messages = build_c1_messages(problem)
    apply_chat = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat):
        raise TypeError(
            "tokenizer must implement apply_chat_template; "
            "refusing to silently fall back (that would hide a train/eval mismatch)"
        )
    base_text = apply_chat(
        messages,
        tokenize=False,
        add_generation_prompt=C1_ADD_GENERATION_PROMPT,
    )
    return f"{base_text}{C1_THINK_PREFILL}"


class C1PromptMixin:
    """
    Mixin: C.1 system + C.1 user + Qwen3-Base chat template + ``<think>\\n``.

    Any adaptor that inherits this gets ``raw_prompts = True``. Override
    ``_get_question_text`` to choose which item field is the problem.

    Optional constructor kwargs (stripped before ``BaseAdaptor.__init__``):

    - ``tokenizer``: object with ``apply_chat_template``
    - ``tokenizer_name``: Hugging Face id, loaded lazily (GPU host)
    """

    raw_prompts = True

    def __init__(self, data_path: str, thinking_mode: bool = False, **kwargs):
        self._tokenizer = kwargs.pop("tokenizer", None)
        self._tokenizer_name = kwargs.pop("tokenizer_name", None)
        super().__init__(data_path, thinking_mode, **kwargs)

    def _get_system_prompt(self) -> str:
        return C1_SYSTEM_PROMPT

    def _get_question_text(self, item: Dict[str, Any]) -> str:
        return item.get("question", "")

    def _ensure_tokenizer(self) -> None:
        if self._tokenizer is not None:
            return
        if self._tokenizer_name:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self._tokenizer_name, trust_remote_code=True
            )
            return
        # CPU / no-weights path: the published Base ChatML render.
        # GPU host must confirm live tokenizers match render_c1_chatml.
        self._tokenizer = _PublishedQwen3BaseChatTemplate()

    def format_prompt(self, item: Dict[str, Any]) -> str:
        self._ensure_tokenizer()
        return render_c1_prompt(self._get_question_text(item), self._tokenizer)


class _PublishedQwen3BaseChatTemplate:
    """
    Stand-in that implements the published Qwen3-Base ``chat_template``
    on the only path this mixin uses (no tools, system+user, generation
    prompt on, ``enable_thinking`` undefined).

    Used when no Hugging Face tokenizer is available so CPU tests can
    freeze the exact rendered string. Not a license to skip GPU
    verification of the live tokenizers.
    """

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    ) -> str:
        if tokenize:
            raise ValueError("this stand-in only supports tokenize=False")
        if kwargs.get("tools"):
            raise ValueError("C.1 path does not use tools")
        if "enable_thinking" in kwargs:
            raise ValueError(
                "C.1 mixin must not pass enable_thinking; "
                "enable_thinking=False would close the think block"
            )
        if not messages or messages[0].get("role") != "system":
            raise ValueError("C.1 messages must start with the system turn")
        parts = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)
