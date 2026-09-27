"""E0 state sampling, per-bin rates, and the 5-minute dry-run budget."""

from __future__ import annotations

import json

from opd_eval.e0 import (
    DRY_RUN_MAX_SECONDS,
    dry_run_limits,
    estimate_seconds,
    full_job_estimate,
    run_e0,
    state_positions,
)


def test_state_positions_are_inside_the_rollout() -> None:
    assert state_positions(1, 4) == []
    assert state_positions(10, 1) == [5]
    positions = state_positions(100, 4)
    assert len(positions) == 4
    assert positions[0] >= 1
    assert positions[-1] <= 99
    assert positions == sorted(set(positions))


def test_continuations_are_verified_per_bin() -> None:
    samples = [
        {"problem_id": "a", "sample_idx": 0, "verified": False, "token_ids": list(range(20)), "ground_truth": "1"},
        {"problem_id": "a", "sample_idx": 1, "verified": True, "token_ids": list(range(20)), "ground_truth": "1"},
        {"problem_id": "b", "sample_idx": 0, "verified": False, "token_ids": list(range(20)), "ground_truth": "1"},
        {"problem_id": "c", "sample_idx": 0, "verified": False, "token_ids": list(range(20)), "ground_truth": "1"},
    ]
    bins = {"a": "B0", "b": "B1", "c": "B3"}

    def generate(state):
        # Position 10 (the single state when n_states=1 and n=20 → 10) succeeds for B0 only.
        if state["bin"] == "B0":
            return ["\\boxed{1}"] * state["n"]
        return ["nope"] * state["n"]

    def verify(text, gt):
        return gt in text

    report = run_e0(
        samples,
        bins,
        generate_fn=generate,
        verify_fn=verify,
        n_states=1,
        n_rollouts=1,
    )
    assert "B3" not in report["bins"]
    assert report["bins"]["B0"]["mean_success"] == 1.0
    assert report["bins"]["B1"]["mean_success"] == 0.0
    assert report["n_generations"] == 8  # 2 problems × 1 state × 4
    assert report["temperature"] == 0.6
    assert report["top_p"] == 0.95


def test_dry_run_budget_is_under_five_minutes() -> None:
    limits = dry_run_limits(2.75)
    n_gen = limits["n_problems"] * limits["n_rollouts"] * limits["n_states"] * limits["n_continuations"]
    assert estimate_seconds(n_gen, 2.75) <= DRY_RUN_MAX_SECONDS
    # H-mini B0+B1 is 100 problems. 4 states, 1 failed rollout, 4 continuations.
    full = full_job_estimate(100, n_states=4, samples_per_sec=2.75)
    assert full["n_generations"] == 100 * 4 * 4
    assert abs(full["estimated_seconds"] - (1600 / 2.75)) < 1e-9


def test_cli_dry_run(tmp_path, capsys) -> None:
    from opd_eval.e0 import main

    out = tmp_path / "e0.json"
    assert main(["--dry-run", "--n-states", "8", "--out", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["dry_run"] is True
    assert payload["estimated_seconds"] <= DRY_RUN_MAX_SECONDS
    assert payload["full_job"]["n_generations"] == 2 * 1 * 8 * 4
    assert capsys.readouterr().out == ""
