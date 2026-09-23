"""Frozen length budget (proposal v4.3 / 2026-09-24 decisions).

Student eval, validation, and binning share:

    prompt ≤ 1024 + response ≤ 10240  ⇒  max_model_len = 11264

Teacher acceptance stays ``|T| ≤ 8192`` (natural EOS).
"""

from __future__ import annotations

PROMPT_MAX_TOKENS = 1024
STUDENT_MAX_NEW_TOKENS = 10240
MAX_MODEL_LEN = PROMPT_MAX_TOKENS + STUDENT_MAX_NEW_TOKENS  # 11264

# Teacher-trace acceptance (|T| ≤ 8192, natural EOS).
TEACHER_ACCEPT_MAX_TOKENS = 8192
# Binning / pass@16 student cap (aligned with paper eval response budget).
CORPUS_PASS16_MAX_NEW_TOKENS = 10240
