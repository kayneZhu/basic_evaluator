"""Output layout: INTERFACE.md paths plus a protocol-profile directory.

INTERFACE.md writes

    eval/outputs/{run_id}/{student_slug}/global_step_{step}/{benchmark_id}/samples.jsonl

The 2026-09-22 roster decision allows the same ``benchmark_id`` on two §5.1
surfaces (e.g. SciBench at pass@1 in the roster and again on the harm
surface). Those runs collide if the path is only ``benchmark_id``. The
extra directory is ``{surface}_k{K}_t{T}``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union


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
