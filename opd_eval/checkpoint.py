"""
Checkpoint reader for INTERFACE.md §1.

Training writes ``…/global_step_{step}/hf/`` plus a sidecar ``manifest.json``.
Eval loads **only** ``hf/`` via transformers. The run is refused if ``g_eval``
is missing or nonzero — evaluation is always g=0.

``C00`` / ``C00p`` do not train and have no ``train/outputs/`` directory.
They resolve to the public Base snapshot plus an eval seed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional


PUBLIC_BASE_BY_SLUG = {
    "qwen3-1.7b-base": "Qwen/Qwen3-1.7B-Base",
    "qwen3-0.6b-base": "Qwen/Qwen3-0.6B-Base",
}

BASE_RUN_IDS = frozenset({"C00", "C00p"})


class CheckpointRefused(ValueError):
    """Checkpoint is not legal for paper evaluation."""


@dataclass(frozen=True)
class CheckpointSpec:
    run_id: str
    student_hf_id: str
    student_slug: str
    teacher_hf_id: Optional[str]
    step: int
    g_eval: int
    model_path: str
    manifest_path: str
    eval_seed: Optional[int] = None
    is_public_base: bool = False


def _require(data: Mapping[str, Any], key: str) -> Any:
    if key not in data:
        raise CheckpointRefused(f"manifest missing required field {key!r}")
    return data[key]


def read_checkpoint(ckpt_dir: Path) -> CheckpointSpec:
    """
    Read ``manifest.json`` beside ``hf/`` and refuse unless ``g_eval == 0``.

    ``ckpt_dir`` is the ``global_step_{step}/`` directory, not ``hf/``.
    """
    ckpt_dir = Path(ckpt_dir)
    manifest_path = ckpt_dir / "manifest.json"
    hf_dir = ckpt_dir / "hf"
    if not manifest_path.is_file():
        raise CheckpointRefused(
            f"missing manifest.json beside hf/ under {ckpt_dir}"
        )
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CheckpointRefused(f"manifest.json is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise CheckpointRefused("manifest.json must be an object")

    if "g_eval" not in data:
        raise CheckpointRefused(
            "g_eval is missing; evaluation refuses the checkpoint "
            "(every paper eval is g=0)"
        )
    g_eval = data["g_eval"]
    if g_eval != 0:
        raise CheckpointRefused(
            f"g_eval={g_eval!r} is nonzero; evaluation refuses the checkpoint "
            "(every paper eval is g=0)"
        )

    if not hf_dir.is_dir():
        raise CheckpointRefused(
            f"eval loads only hf/ via transformers; missing directory {hf_dir}"
        )

    return CheckpointSpec(
        run_id=str(_require(data, "run_id")),
        student_hf_id=str(_require(data, "student_hf_id")),
        student_slug=str(_require(data, "student_slug")),
        teacher_hf_id=(
            None if data.get("teacher_hf_id") is None
            else str(data["teacher_hf_id"])
        ),
        step=int(_require(data, "step")),
        g_eval=0,
        model_path=str(hf_dir),
        manifest_path=str(manifest_path),
        is_public_base=False,
    )


def resolve_public_base(
    run_id: str,
    student_slug: str,
    *,
    eval_seed: Optional[int] = None,
    student_hf_id: Optional[str] = None,
) -> CheckpointSpec:
    if run_id not in BASE_RUN_IDS:
        raise CheckpointRefused(
            f"{run_id} is not a public-Base run (C00 / C00p)"
        )
    hf_id = student_hf_id or PUBLIC_BASE_BY_SLUG.get(student_slug)
    if not hf_id:
        raise CheckpointRefused(
            f"unknown student_slug {student_slug!r}; expected one of "
            f"{sorted(PUBLIC_BASE_BY_SLUG)}"
        )
    return CheckpointSpec(
        run_id=run_id,
        student_hf_id=hf_id,
        student_slug=student_slug,
        teacher_hf_id=None,
        step=0,
        g_eval=0,
        model_path=hf_id,
        manifest_path="",
        eval_seed=eval_seed,
        is_public_base=True,
    )


def resolve_eval_target(
    run_id: str,
    student_slug: str,
    *,
    ckpt_dir: Optional[Path] = None,
    eval_seed: Optional[int] = None,
    student_hf_id: Optional[str] = None,
) -> CheckpointSpec:
    if run_id in BASE_RUN_IDS:
        return resolve_public_base(
            run_id,
            student_slug,
            eval_seed=eval_seed,
            student_hf_id=student_hf_id,
        )
    if ckpt_dir is None:
        raise CheckpointRefused(
            f"{run_id} requires a trained checkpoint directory with hf/ + manifest.json"
        )
    spec = read_checkpoint(ckpt_dir)
    if spec.run_id != run_id:
        raise CheckpointRefused(
            f"manifest run_id={spec.run_id!r} does not match requested {run_id!r}"
        )
    if spec.student_slug != student_slug:
        raise CheckpointRefused(
            f"manifest student_slug={spec.student_slug!r} does not match "
            f"requested {student_slug!r}"
        )
    return spec
