"""Frozen length budget (proposal v4.3 / 2026-09-24 decisions).

Student eval, validation, and binning share (defaults):

    prompt ≤ 1024 + response ≤ 10240  ⇒  max_model_len = 11264

Path A may raise the response cap (e.g. 13k/16k) via CLI ``--max-new-tokens``
or env ``OPD_EVAL_MAX_NEW_TOKENS``; ``max_model_len`` must be
``PROMPT_MAX_TOKENS + max_new_tokens`` (see ``resolve_eval_length_budget``).

Teacher acceptance stays ``|T| ≤ 8192`` (natural EOS) unless a later D_t
filter chooses a tighter cap (e.g. 7168).
"""

from __future__ import annotations

import os
from typing import Optional

PROMPT_MAX_TOKENS = 1024
STUDENT_MAX_NEW_TOKENS = 10240
MAX_MODEL_LEN = PROMPT_MAX_TOKENS + STUDENT_MAX_NEW_TOKENS  # 11264

# Teacher-trace acceptance (|T| ≤ 8192, natural EOS).
TEACHER_ACCEPT_MAX_TOKENS = 8192
# Binning / pass@16 student cap (aligned with paper eval response budget).
CORPUS_PASS16_MAX_NEW_TOKENS = 10240


def resolve_eval_max_new_tokens(cli_value: Optional[int] = None) -> int:
    """CLI wins; else ``OPD_EVAL_MAX_NEW_TOKENS``; else the frozen default."""
    if cli_value is not None:
        return int(cli_value)
    env = os.environ.get("OPD_EVAL_MAX_NEW_TOKENS", "").strip()
    if env:
        return int(env)
    return int(STUDENT_MAX_NEW_TOKENS)


def resolve_eval_max_model_len(
    max_new_tokens: int,
    *,
    cli_max_model_len: Optional[int] = None,
) -> int:
    """Default ``prompt + response``; explicit CLI max_model_len wins when set."""
    if cli_max_model_len is not None:
        return int(cli_max_model_len)
    return int(PROMPT_MAX_TOKENS) + int(max_new_tokens)


def resolve_eval_model_dir(cli_value: Optional[str] = None) -> Optional[str]:
    """CLI ``--model-dir`` wins; else ``OPD_EVAL_MODEL_DIR`` / ``OPD_STUDENT_MODEL``."""
    if cli_value:
        return str(cli_value)
    for key in ("OPD_EVAL_MODEL_DIR", "OPD_STUDENT_MODEL"):
        env = os.environ.get(key, "").strip()
        if env:
            return env
    return None
