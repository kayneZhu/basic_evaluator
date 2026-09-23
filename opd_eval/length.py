"""Frozen length budget (proposal v4.3 / 2026-09-23 decisions).

Student eval and training share:

    prompt ≤ 1024 + response ≤ 10240  ⇒  max_model_len = 11264

Teacher acceptance stays ``|T| ≤ 8192`` (natural EOS). Existing U1 / corpus
pass@16 shards written under the old 8192 student cap are **valid and
reusable** — evaluation must not reject them or require a re-run.
"""

from __future__ import annotations

PROMPT_MAX_TOKENS = 1024
STUDENT_MAX_NEW_TOKENS = 10240
MAX_MODEL_LEN = PROMPT_MAX_TOKENS + STUDENT_MAX_NEW_TOKENS  # 11264

# Teacher-trace acceptance and historical corpus pass@16 (U1/U4) cap.
TEACHER_ACCEPT_MAX_TOKENS = 8192
CORPUS_PASS16_MAX_NEW_TOKENS = 8192
