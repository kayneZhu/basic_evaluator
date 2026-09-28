"""Arbitrary --extra-set jsonl uses the same H-mini prompt + scorer path."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptors.adaptor_factory import AdaptorFactory
from opd_eval.full_protocol import (
    H_MINI_N,
    build_extra_jobs,
    build_h_mini_n512_job,
    parse_extra_set_spec,
)

ROOT = Path(__file__).resolve().parents[1]
H_MINI = ROOT / "data" / "processed" / "h_mini200.jsonl"
EXTRA50 = ROOT / "data" / "h400_b0_extra50.jsonl"


def test_parse_extra_set_spec() -> None:
    name, path = parse_extra_set_spec("h_b0_extra=/tmp/x.jsonl")
    assert name == "h_b0_extra"
    assert path == Path("/tmp/x.jsonl")
    with pytest.raises(ValueError):
        parse_extra_set_spec("missing_eq")


def test_build_extra_jobs_match_h_mini_contract() -> None:
    jobs = build_extra_jobs(["h_b0_extra=/tmp/x.jsonl"], n=512)
    assert len(jobs) == 1
    job = jobs[0]
    h_mini = build_h_mini_n512_job(Path("/tmp/h_mini200.jsonl"))
    assert job.n == h_mini.n == H_MINI_N
    assert job.adaptor_key == h_mini.adaptor_key == "c1_or1_200"
    assert job.surface == "h_b0_extra"
    assert job.benchmark_id == "h_b0_extra"
    assert job.bin_filter is None
    assert job.bin_exclude is None


def test_extra50_file_is_fifty_b0_not_in_h_mini() -> None:
    assert EXTRA50.is_file()
    assert H_MINI.is_file()
    mini_ids = {
        json.loads(line).get("or1_id")
        for line in H_MINI.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    rows = [
        json.loads(line)
        for line in EXTRA50.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 50
    assert all(r.get("bin") == "B0" for r in rows)
    assert all(r.get("or1_id") not in mini_ids for r in rows)


def test_extra_set_prompts_and_scores_match_h_mini_on_three_problems(tmp_path) -> None:
    """Byte-identical C.1 prompts and Math-Verify scores vs the H-mini path."""
    mini_rows = [
        json.loads(line)
        for line in H_MINI.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][:3]
    assert len(mini_rows) == 3
    h_mini_path = tmp_path / "h_mini3.jsonl"
    extra_path = tmp_path / "extra3.jsonl"
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in mini_rows)
    h_mini_path.write_text(text, encoding="utf-8")
    extra_path.write_text(text, encoding="utf-8")

    h_job = build_h_mini_n512_job(h_mini_path)
    e_job = build_extra_jobs([f"extra3={extra_path}"], n=512)[0]
    assert h_job.adaptor_key == e_job.adaptor_key

    h_ad = AdaptorFactory.create_adaptor(h_job.adaptor_key, str(h_job.data_path), thinking_mode=False)
    e_ad = AdaptorFactory.create_adaptor(e_job.adaptor_key, str(e_job.data_path), thinking_mode=False)
    assert len(h_ad.data) == len(e_ad.data) == 3

    for i, (h_item, e_item) in enumerate(zip(h_ad.data, e_ad.data)):
        assert h_ad.get_problem_id(h_item) == e_ad.get_problem_id(e_item)
        h_prompt = h_ad.format_prompt(h_item)
        e_prompt = e_ad.format_prompt(e_item)
        assert h_prompt == e_prompt
        assert isinstance(h_prompt, str) and len(h_prompt) > 0
        # Byte identity of the rendered ChatML bytes (CPU published template).
        assert h_prompt.encode("utf-8") == e_prompt.encode("utf-8")
        gt = h_ad.get_ground_truth(h_item)
        assert gt == e_ad.get_ground_truth(e_item)
        # One correct boxed answer (string-equal to GT) and one unparseable.
        responses = [
            f"reasoning done.\n\\boxed{{{gt}}}",
            "no final answer",
        ]
        for response in responses:
            h_ext = h_ad.extract_answer(response)
            e_ext = e_ad.extract_answer(response)
            assert h_ext == e_ext
            h_ok = h_ad.verify_answer(h_ext, gt)
            e_ok = e_ad.verify_answer(e_ext, gt)
            assert h_ok == e_ok
            if response.startswith("reasoning"):
                assert h_ok is True
            else:
                assert h_ok is False


def test_cli_dry_plan_extra_only(tmp_path) -> None:
    from opd_eval.full_protocol import main

    path = tmp_path / "toy.jsonl"
    path.write_text(
        json.dumps(
            {
                "or1_id": "or1:1",
                "problem_id": "or1:1",
                "index": 1,
                "bin": "B0",
                "question": "1+1?",
                "ground_truth": "2",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rc = main(
        [
            "--model-dir",
            str(tmp_path / "model"),
            "--out-root",
            str(tmp_path / "out"),
            "--dry-plan",
            "--only-extra",
            "--extra-set",
            f"toy={path}",
            "--extra-n",
            "512",
        ]
    )
    assert rc == 0
