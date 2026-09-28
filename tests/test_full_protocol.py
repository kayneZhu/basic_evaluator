"""Full-set plan: eight pass@1 columns, Fig 4 sampled once at n=512."""

from __future__ import annotations

import pytest

from opd_eval.full_protocol import (
    FIG4_N,
    FULL_PROTOCOL_SEED,
    PASS1_N,
    build_full_jobs,
    plan_summary,
)


def test_fig4_surfaces_are_n512_and_not_resampled_at_16() -> None:
    jobs = build_full_jobs()
    by_surface = {j.surface: j for j in jobs}
    assert set(by_surface) == {"h", "h_hard", "math500", "aime_union", "amc23", "hmmt25"}
    for surface in ("h_hard", "math500", "aime_union"):
        assert by_surface[surface].n == FIG4_N
    for surface in ("h", "amc23", "hmmt25"):
        assert by_surface[surface].n == PASS1_N
    assert by_surface["h_hard"].bin_filter == "B0"
    summary = plan_summary(jobs, base_seed=FULL_PROTOCOL_SEED)
    assert summary["n_seeds"] == 1
    assert summary["pass1_from_n512"] == ["h_hard", "math500", "aime_union"]
    assert len(summary["benchmarks_pass1"]) == 8
    assert "h_k128" not in by_surface


def test_h_mini_n512_is_opt_in_and_keeps_every_sample() -> None:
    from pathlib import Path

    from opd_eval.full_protocol import H_MINI_N, build_extra_jobs, build_h_mini_n512_job

    assert "h_mini" not in {j.surface for j in build_full_jobs()}
    job = build_h_mini_n512_job(Path("/tmp/h_mini200.jsonl"))
    assert job.surface == "h_mini"
    assert job.benchmark_id == "h_mini"
    assert job.n == H_MINI_N == 512
    assert job.bin_filter is None
    assert job.bin_exclude is None
    summary = plan_summary([job], base_seed=FULL_PROTOCOL_SEED)
    assert summary["jobs"][0]["n"] == 512
    assert summary["fig4_n512"] == ["h_mini"]
    extra = build_extra_jobs(["h_b0_extra=/tmp/extra.jsonl"], n=512)
    combined = plan_summary([job, *extra], base_seed=FULL_PROTOCOL_SEED)
    assert [j["surface"] for j in combined["jobs"]] == ["h_mini", "h_b0_extra"]


def test_h_k128_is_opt_in() -> None:
    jobs = build_full_jobs(include_h_k128=True)
    k128 = [j for j in jobs if j.surface == "h_k128"]
    assert len(k128) == 1 and k128[0].n == 128


def test_seed_zero_is_refused() -> None:
    jobs = build_full_jobs()
    with pytest.raises(ValueError):
        plan_summary(jobs, base_seed=0)
