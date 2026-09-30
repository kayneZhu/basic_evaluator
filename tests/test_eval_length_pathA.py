"""Path A eval length / model-dir resolution."""

from __future__ import annotations

import pytest

from opd_eval.length import (
    PROMPT_MAX_TOKENS,
    STUDENT_MAX_NEW_TOKENS,
    resolve_eval_max_model_len,
    resolve_eval_max_new_tokens,
    resolve_eval_model_dir,
)


def test_resolve_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPD_EVAL_MAX_NEW_TOKENS", raising=False)
    assert resolve_eval_max_new_tokens(None) == STUDENT_MAX_NEW_TOKENS
    assert resolve_eval_max_new_tokens(16384) == 16384
    monkeypatch.setenv("OPD_EVAL_MAX_NEW_TOKENS", "13000")
    assert resolve_eval_max_new_tokens(None) == 13000
    assert resolve_eval_max_model_len(16384) == PROMPT_MAX_TOKENS + 16384
    assert resolve_eval_max_model_len(16384, cli_max_model_len=20000) == 20000


def test_resolve_model_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPD_EVAL_MODEL_DIR", raising=False)
    monkeypatch.delenv("OPD_STUDENT_MODEL", raising=False)
    assert resolve_eval_model_dir(None) is None
    assert resolve_eval_model_dir("/tmp/m") == "/tmp/m"
    monkeypatch.setenv("OPD_STUDENT_MODEL", "/root/autodl-tmp/models/Qwen/Qwen3-1.7B")
    assert resolve_eval_model_dir(None).endswith("Qwen3-1.7B")
