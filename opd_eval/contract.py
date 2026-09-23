"""Cross-repo generation contract: g=0 prompt + EOS / stop convention.

Authority is ``contract/g0_prompt.golden.json`` in this repo. Token ids
live in this same fixture, not a sibling file: the prompt bytes and the
generation-stop convention are one frozen contract. A sibling would be
a second load path that can be bumped independently of the prompt it
must agree with.

``sha256`` remains the template-byte digest (the string is frozen).
``stop_sha256`` hashes the terminator, stop set, and the Base default
being overridden.

The student is trained to emit the terminator because T(x) ends there,
so stopping only on the Base default would mean the student never
terminates and every sample runs to the full student response budget
(10240 tokens); conversely a Base model can spontaneously emit the
default, so including it in the stop set costs nothing and prevents the
same waste. Do not read ``tokenizer.eos_token_id`` anywhere on this path.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

EVAL_ROOT = Path(__file__).resolve().parents[1]
G0_GOLDEN_JSON = EVAL_ROOT / "contract" / "g0_prompt.golden.json"
_PLACEHOLDER = "{problem}"


def _require_prompt_fields(payload: dict[str, Any]) -> None:
    template = payload.get("template")
    if not isinstance(template, str) or _PLACEHOLDER not in template:
        raise ValueError(
            "g=0 golden JSON is missing a template with {problem}; "
            "refusing to invent a golden string"
        )
    if "<think>" not in template:
        raise ValueError("golden template is missing the format-only <think> prefill")


def _require_stop_fields(payload: dict[str, Any]) -> None:
    terminator = payload.get("trace_terminator_token_id")
    stops = payload.get("stop_token_ids")
    default = payload.get("base_default_eos_token_id")
    digest = payload.get("stop_sha256")
    if not isinstance(terminator, int):
        raise ValueError(
            "g=0 golden JSON is missing trace_terminator_token_id; "
            "refusing to invent a terminator"
        )
    if not (isinstance(stops, list) and stops and all(isinstance(x, int) for x in stops)):
        raise ValueError(
            "g=0 golden JSON is missing stop_token_ids; "
            "refusing to invent a stop set"
        )
    if not isinstance(default, int):
        raise ValueError(
            "g=0 golden JSON is missing base_default_eos_token_id; "
            "refusing to invent the overridden tokenizer default"
        )
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(
            "g=0 golden JSON is missing stop_sha256; "
            "refusing an unhashed stop convention"
        )


def load_g0_golden() -> dict[str, Any]:
    if not G0_GOLDEN_JSON.is_file():
        raise FileNotFoundError(
            f"missing generation contract {G0_GOLDEN_JSON}; "
            "refusing to invent a golden string or stop set"
        )
    payload = json.loads(G0_GOLDEN_JSON.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("g=0 golden JSON must be an object")
    _require_prompt_fields(payload)
    _require_stop_fields(payload)
    return payload


def golden_g0_prompt(problem: str) -> str:
    return load_g0_golden()["template"].replace(_PLACEHOLDER, problem, 1)


def template_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(payload["template"].encode("utf-8")).hexdigest()


def stop_convention_digest(payload: dict[str, Any]) -> str:
    """sha256 of the stop convention, matching train's drift guard.

    Canonical form is sorted-key JSON of the three integer fields so a
    pretty-print change of the fixture cannot hide a token-id edit.
    """
    canonical = {
        "base_default_eos_token_id": int(payload["base_default_eos_token_id"]),
        "stop_token_ids": [int(x) for x in payload["stop_token_ids"]],
        "trace_terminator_token_id": int(payload["trace_terminator_token_id"]),
    }
    blob = json.dumps(canonical, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    return hashlib.sha256(blob).hexdigest()


def trace_terminator_token_id() -> int:
    """T(x) terminator. Loaded from the contract; not a local literal."""
    return int(load_g0_golden()["trace_terminator_token_id"])


def stop_token_ids() -> tuple[int, ...]:
    """Generation stop set. Loaded from the contract; not a local literal."""
    return tuple(int(x) for x in load_g0_golden()["stop_token_ids"])


def base_default_eos_token_id() -> int:
    """Base ``eos_token_id`` that the stop set overrides."""
    return int(load_g0_golden()["base_default_eos_token_id"])


def vllm_stop_token_ids() -> list[int]:
    return list(stop_token_ids())


def vllm_sampling_kwargs(
    *,
    max_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    stop: Optional[list[str]] = None,
    n: int = 1,
) -> dict[str, Any]:
    """vLLM SamplingParams kwargs. Always includes the contract stop set."""
    return {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stop": stop,
        "n": n,
        "stop_token_ids": vllm_stop_token_ids(),
    }
