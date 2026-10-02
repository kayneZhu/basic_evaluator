"""E0: success of student continuations from states on failed rollouts.

Uses the Base 1.7B H-mini n=512 samples (``samples.jsonl`` under
``eval_v5_hmini200_20260927/base_1p7b``). Low-p bins are B0 and B1.
Each selected failed rollout contributes ``n_states`` token positions.
Each state is continued ``n_continuations`` times (default 4) at
temperature 0.6 / top_p 0.95. The per-bin output is the distribution of
per-state success rates.

Runtime at ``samples_per_sec`` (measured 2.75 samples/s on 4×5090):

    n_gen = n_problems * n_rollouts * n_states * n_continuations
    seconds = n_gen / samples_per_sec

``--dry-run`` executes at most a 5-minute slice of that budget (stub
generator, no vLLM) and still prints the full-job estimate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

LOW_BINS = ("B0", "B1")
N_CONTINUATIONS = 4
TEMPERATURE = 0.6
TOP_P = 0.95
SAMPLES_PER_SEC = 2.75
DRY_RUN_MAX_SECONDS = 300.0

GenerateFn = Callable[[Mapping[str, Any]], Sequence[str]]
VerifyFn = Callable[[str, str], bool]


def estimate_seconds(n_generations: int, samples_per_sec: float = SAMPLES_PER_SEC) -> float:
    if samples_per_sec <= 0:
        raise ValueError("samples_per_sec must be positive")
    return float(n_generations) / float(samples_per_sec)


def state_positions(n_tokens: int, n_states: int) -> list[int]:
    """Token prefixes in ``1 .. n_tokens-1``. Evenly spaced, stable."""
    if n_states <= 0 or n_tokens <= 1:
        return []
    last = n_tokens - 1
    if n_states == 1:
        return [max(1, n_tokens // 2)]
    if n_states >= last:
        return list(range(1, n_tokens))
    raw = [1 + round(i * (last - 1) / (n_states - 1)) for i in range(n_states)]
    out: list[int] = []
    for pos in raw:
        pos = max(1, min(last, int(pos)))
        if pos not in out:
            out.append(pos)
    return out


def dry_run_limits(samples_per_sec: float = SAMPLES_PER_SEC) -> dict[str, int]:
    """A slice whose estimated wall time is at most five minutes."""
    # 2 problems × 1 rollout × 1 state × 4 continuations.
    n_gen = 2 * 1 * 1 * N_CONTINUATIONS
    if estimate_seconds(n_gen, samples_per_sec) > DRY_RUN_MAX_SECONDS:
        raise RuntimeError("dry-run budget exceeds 5 minutes; shrink the slice")
    return {"n_problems": 2, "n_rollouts": 1, "n_states": 1, "n_continuations": N_CONTINUATIONS}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _problem_id(row: Mapping[str, Any]) -> str:
    return str(row.get("problem_id") or row.get("or1_id") or "")


def load_bin_map(h_mini_path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in _read_jsonl(h_mini_path):
        pid = _problem_id(row)
        if pid:
            out[pid] = str(row.get("bin") or "")
    return out


def load_problem_texts(h_mini_path: Path) -> dict[str, str]:
    """Problem string from the H-mini / seat manifest, keyed by problem id.

    Held-out H stores that string in ``question`` (``sample_heldout_h``
    copies ``problem.problem_text``). Seated rows store it in
    ``problem_text``. Never read it from ``samples.jsonl``.
    """
    out: dict[str, str] = {}
    for row in _read_jsonl(h_mini_path):
        pid = _problem_id(row)
        if pid:
            out[pid] = str(row.get("problem_text") or row.get("question") or "")
    return out


def problem_text_for(problem_id: str, problems: Mapping[str, str] | None) -> str:
    text = "" if problems is None else str(problems.get(problem_id) or "")
    if not text.strip():
        raise ValueError(
            f"E0 missing problem_text for {problem_id} on the H-mini / seat manifest"
        )
    return text


def select_failed_rollouts(
    samples: Sequence[Mapping[str, Any]],
    bins: Mapping[str, str],
    *,
    n_rollouts: int,
    low_bins: Sequence[str] = LOW_BINS,
    problem_limit: int | None = None,
) -> list[dict[str, Any]]:
    """Up to ``n_rollouts`` failed samples per problem, lowest ``sample_idx`` first."""
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in samples:
        if bool(row.get("verified")):
            continue
        pid = _problem_id(row)
        if bins.get(pid) not in low_bins:
            continue
        grouped.setdefault(pid, []).append(row)
    pids = sorted(grouped)
    if problem_limit is not None:
        pids = pids[: int(problem_limit)]
    chosen: list[dict[str, Any]] = []
    for pid in pids:
        rows = sorted(grouped[pid], key=lambda r: int(r.get("sample_idx") or 0))
        for row in rows[: int(n_rollouts)]:
            item = dict(row)
            item["bin"] = bins[pid]
            chosen.append(item)
    return chosen


def response_token_ids(row: Mapping[str, Any], tokenizer: Any) -> list[int]:
    """Student-tokenizer ids of the rollout text. No invented ``range(n)``."""
    if tokenizer is None:
        raise ValueError("E0 needs the student tokenizer to retokenize response")
    text = row.get("response")
    if text is None:
        raise ValueError("E0 sample is missing response text")
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise TypeError("student tokenizer must implement encode")
    return [int(x) for x in encode(str(text), add_special_tokens=False)]


def eval_chat_template_ids(question: str, tokenizer: Any) -> list[int]:
    """Eval prompt ids for the selected format (C.1 or C.2). No BOS."""
    from adaptors.c1_prompt_mixin import render_c1_prompt
    from adaptors.prompt_format import (
        PROMPT_FORMAT_C2,
        render_c2_prompt,
        resolve_prompt_format,
    )

    if resolve_prompt_format() == PROMPT_FORMAT_C2:
        text = render_c2_prompt(str(question))
    else:
        text = render_c1_prompt(str(question), tokenizer)
    return [int(x) for x in tokenizer.encode(text, add_special_tokens=False)]


def continuation_prompt_ids(
    question: str,
    prefix_token_ids: Sequence[int],
    tokenizer: Any,
) -> list[int]:
    """Eval chat-template ids followed by the failed-rollout prefix."""
    return eval_chat_template_ids(question, tokenizer) + [int(x) for x in prefix_token_ids]


def run_e0(
    samples: Sequence[Mapping[str, Any]],
    bins: Mapping[str, str],
    *,
    generate_fn: GenerateFn,
    verify_fn: VerifyFn,
    n_states: int,
    n_rollouts: int = 1,
    n_continuations: int = N_CONTINUATIONS,
    problem_limit: int | None = None,
    samples_per_sec: float = SAMPLES_PER_SEC,
    answers: Mapping[str, str] | None = None,
    tokenizer: Any = None,
    problems: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Score continuations. ``generate_fn(state)`` returns ``n`` new suffixes."""
    rollouts = select_failed_rollouts(
        samples, bins, n_rollouts=n_rollouts, problem_limit=problem_limit
    )
    per_bin_rates: dict[str, list[float]] = {name: [] for name in LOW_BINS}
    n_gen = 0
    details: list[dict[str, Any]] = []
    for row in rollouts:
        ids = response_token_ids(row, tokenizer)
        positions = state_positions(len(ids), n_states)
        gt = ""
        if answers is not None:
            gt = str(answers.get(_problem_id(row), ""))
        gt = gt or str(row.get("ground_truth") or row.get("answer") or "")
        question = problem_text_for(_problem_id(row), problems)
        for pos in positions:
            prefix = ids[:pos]
            state = {
                "problem_id": _problem_id(row),
                "bin": row["bin"],
                "sample_idx": int(row.get("sample_idx") or 0),
                "position": int(pos),
                "prefix_token_ids": prefix,
                "prompt_token_ids": continuation_prompt_ids(question, prefix, tokenizer),
                "question": question,
                "ground_truth": gt,
                "n": int(n_continuations),
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
            }
            continuations = list(generate_fn(state))
            if len(continuations) != int(n_continuations):
                raise RuntimeError(
                    f"generate_fn returned {len(continuations)} strings, want {n_continuations}"
                )
            n_gen += len(continuations)
            hits = sum(1 for text in continuations if verify_fn(str(text), gt))
            rate = hits / float(n_continuations)
            per_bin_rates.setdefault(str(row["bin"]), []).append(rate)
            details.append(
                {
                    "problem_id": state["problem_id"],
                    "bin": state["bin"],
                    "sample_idx": state["sample_idx"],
                    "position": state["position"],
                    "n_hits": hits,
                    "n": int(n_continuations),
                    "success_rate": rate,
                }
            )
    bins_out: dict[str, Any] = {}
    for name, rates in per_bin_rates.items():
        if not rates:
            continue
        bins_out[name] = {
            "n_states": len(rates),
            "mean_success": sum(rates) / len(rates),
            "rates": rates,
        }
    return {
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "n_states": int(n_states),
        "n_rollouts": int(n_rollouts),
        "n_continuations": int(n_continuations),
        "n_generations": n_gen,
        "samples_per_sec": float(samples_per_sec),
        "estimated_seconds": estimate_seconds(n_gen, samples_per_sec),
        "bins": bins_out,
        "states": details,
    }


def full_job_estimate(
    n_problems: int,
    *,
    n_states: int,
    n_rollouts: int = 1,
    n_continuations: int = N_CONTINUATIONS,
    samples_per_sec: float = SAMPLES_PER_SEC,
) -> dict[str, float | int]:
    n_gen = int(n_problems) * int(n_rollouts) * int(n_states) * int(n_continuations)
    return {
        "n_problems": int(n_problems),
        "n_generations": n_gen,
        "samples_per_sec": float(samples_per_sec),
        "estimated_seconds": estimate_seconds(n_gen, samples_per_sec),
        "gpus": "4x5090",
    }


def _stub_generate(state: Mapping[str, Any]) -> list[str]:
    return ["unsolved"] * int(state["n"])


def vllm_generate_fn(model: Path, *, temperature: float = TEMPERATURE, top_p: float = TOP_P):
    """Student continuations at T=0.6 / top_p=0.95. One engine, ``n`` samples per state.

    The prompt is ``state["prompt_token_ids"]``: eval chat-template ids
    plus the retokenized failed-rollout prefix.
    """
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(model),
        tensor_parallel_size=1,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
    )
    tokenizer = llm.get_tokenizer()

    def _gen(state: Mapping[str, Any]) -> list[str]:
        prompt_ids = list(state["prompt_token_ids"])
        params = SamplingParams(
            temperature=float(temperature),
            top_p=float(top_p),
            n=int(state["n"]),
            max_tokens=10240,
        )
        from vllm.inputs import TokensPrompt

        out = llm.generate(
            [TokensPrompt(prompt_token_ids=prompt_ids)],
            params,
        )[0]
        texts = []
        for sample in out.outputs:
            texts.append(tokenizer.decode(sample.token_ids, skip_special_tokens=True))
        return texts

    _gen.tokenizer = tokenizer  # type: ignore[attr-defined]
    return _gen


def _default_verify(text: str, ground_truth: str) -> bool:
    from adaptors.verl_aligned_adaptor import _extract_boxed, _verify_math

    return bool(_verify_math(_extract_boxed(text), ground_truth))


class _CharTokenizer:
    """Dry-run stand-in. ``--model`` uses the student tokenizer instead."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if add_special_tokens:
            raise ValueError("E0 retokenizes with add_special_tokens=False")
        return [ord(ch) for ch in str(text)]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        if tokenize:
            raise ValueError("render the string, then encode")
        if "enable_thinking" in kwargs:
            raise ValueError("C.1 must not pass enable_thinking")
        parts = [
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            for message in messages
        ]
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, help="Base H-mini samples.jsonl")
    parser.add_argument(
        "--h-mini",
        type=Path,
        help="h_mini200.jsonl (problem_id, bin, answer, problem_text)",
    )
    parser.add_argument("--n-states", type=int, default=4)
    parser.add_argument("--n-rollouts", type=int, default=1)
    parser.add_argument("--samples-per-sec", type=float, default=SAMPLES_PER_SEC)
    parser.add_argument("--dry-run", action="store_true", help="≤5 min stub; no vLLM")
    parser.add_argument("--model", type=Path, help="Student HF dir. Omit with --dry-run.")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    n_states = int(args.n_states)
    n_rollouts = int(args.n_rollouts)
    problem_limit = None
    if args.dry_run:
        limits = dry_run_limits(args.samples_per_sec)
        n_states = limits["n_states"]
        n_rollouts = limits["n_rollouts"]
        problem_limit = limits["n_problems"]

    if args.samples and args.h_mini:
        samples = _read_jsonl(args.samples)
        bins = load_bin_map(args.h_mini)
        problems = load_problem_texts(args.h_mini)
        answers = {}
        for row in _read_jsonl(args.h_mini):
            pid = _problem_id(row)
            answers[pid] = str(row.get("answer") or row.get("ground_truth") or "")
    else:
        if not args.dry_run:
            raise SystemExit("E0 needs --samples and --h-mini (or --dry-run)")
        samples = [
            {
                "problem_id": "p0",
                "sample_idx": 0,
                "verified": False,
                "response": "x" * 20,
                "question": "q0",
                "ground_truth": "1",
            },
            {
                "problem_id": "p1",
                "sample_idx": 0,
                "verified": False,
                "response": "y" * 20,
                "question": "q1",
                "ground_truth": "1",
            },
            {
                "problem_id": "p2",
                "sample_idx": 0,
                "verified": True,
                "response": "z" * 20,
                "question": "q2",
                "ground_truth": "1",
            },
        ]
        bins = {"p0": "B0", "p1": "B1", "p2": "B3"}
        answers = {"p0": "1", "p1": "1", "p2": "1"}
        problems = {
            "p0": "seated problem zero",
            "p1": "seated problem one",
            "p2": "seated problem two",
        }

    n_problems_full = len({
        _problem_id(r)
        for r in samples
        if not r.get("verified") and bins.get(_problem_id(r)) in LOW_BINS
    })
    tokenizer: Any
    if args.dry_run:
        generate_fn: GenerateFn = _stub_generate
        tokenizer = _CharTokenizer()
    elif args.model:
        generate_fn = vllm_generate_fn(args.model)
        tokenizer = generate_fn.tokenizer  # type: ignore[attr-defined]
    else:
        raise SystemExit("pass --dry-run or --model (student HF dir, T=0.6/top_p=0.95)")
    report = run_e0(
        samples,
        bins,
        generate_fn=generate_fn,
        verify_fn=_default_verify,
        n_states=n_states,
        n_rollouts=n_rollouts,
        problem_limit=problem_limit,
        samples_per_sec=float(args.samples_per_sec),
        answers=answers,
        tokenizer=tokenizer,
        problems=problems,
    )
    report["dry_run"] = bool(args.dry_run)
    report["full_job"] = full_job_estimate(
        n_problems_full,
        n_states=int(args.n_states),
        n_rollouts=int(args.n_rollouts),
        samples_per_sec=float(args.samples_per_sec),
    )
    if args.dry_run and float(report["estimated_seconds"]) > DRY_RUN_MAX_SECONDS:
        raise SystemExit(
            f"dry-run estimate {report['estimated_seconds']:.1f}s exceeds {DRY_RUN_MAX_SECONDS:.0f}s"
        )
    text = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
