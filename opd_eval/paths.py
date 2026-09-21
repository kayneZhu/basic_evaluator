"""Output layout: INTERFACE.md paths plus a protocol-profile directory.

INTERFACE.md §1 (training checkpoints):

    train/examples/{example_id}/runs/{experiment_id}/checkpoints/global_step_{step}/

The retired tree ``train/outputs/{run_id}/{student_slug}/global_step_{step}/``
is illegal. INTERFACE.md §2 (eval samples) still writes

    eval/outputs/{run_id}/{student_slug}/global_step_{step}/{benchmark_id}/samples.jsonl

The 2026-09-22 roster decision allows the same ``benchmark_id`` on two §5.1
surfaces (e.g. SciBench at pass@1 in the roster and again on the harm
surface). Those runs collide if the path is only ``benchmark_id``. The
extra directory is ``{surface}_k{K}_t{T}``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

CHECKPOINT_PATH_TEMPLATE = (
    "examples/{example_id}/runs/{experiment_id}/checkpoints/global_step_{step}"
)
LEGACY_CHECKPOINT_PATH_TEMPLATE = (
    "outputs/{run_id}/{student_slug}/global_step_{step}"
)
LEGACY_STUDENT_SLUGS = frozenset({"qwen3-1.7b-base", "qwen3-0.6b-base"})


class StaleCheckpointPath(ValueError):
    """Path matches the retired ``train/outputs/…`` convention."""


def checkpoint_dir(
    train_root: Union[str, Path],
    example_id: str,
    experiment_id: str,
    step: int,
) -> Path:
    return (
        Path(train_root)
        / "examples"
        / example_id
        / "runs"
        / experiment_id
        / "checkpoints"
        / f"global_step_{int(step)}"
    )


def is_legacy_checkpoint_path(ckpt_dir: Union[str, Path]) -> bool:
    parts = Path(ckpt_dir).parts
    for i, part in enumerate(parts):
        if part != "outputs":
            continue
        if i + 3 >= len(parts):
            continue
        slug, step_dir = parts[i + 2], parts[i + 3]
        if slug in LEGACY_STUDENT_SLUGS and step_dir.startswith("global_step_"):
            return True
    return False


def refuse_legacy_checkpoint_path(ckpt_dir: Union[str, Path]) -> None:
    if is_legacy_checkpoint_path(ckpt_dir):
        raise StaleCheckpointPath(
            f"retired checkpoint path {ckpt_dir}: "
            f"{LEGACY_CHECKPOINT_PATH_TEMPLATE} is superseded by "
            f"{CHECKPOINT_PATH_TEMPLATE}"
        )


def protocol_profile_name(surface: str, k: int, temperature: float) -> str:
    if not surface:
        raise ValueError("surface name is required on the protocol profile")
    if k <= 0:
        raise ValueError(f"K must be positive, got {k}")
    return f"{surface}_k{k}_t{temperature:g}"


def samples_output_dir(
    outputs_root: Union[str, Path],
    run_id: str,
    student_slug: str,
    step: int,
    benchmark_id: str,
    surface: str,
    k: int,
    temperature: float,
) -> Path:
    return (
        Path(outputs_root)
        / run_id
        / student_slug
        / f"global_step_{step}"
        / benchmark_id
        / protocol_profile_name(surface, k, temperature)
    )
