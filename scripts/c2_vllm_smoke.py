"""Tiny C.2 vLLM smoke: 4 prompts x one model per process.

Prints the fixed prompt ids and, per sample, the terminal token id,
length, think-tag flags, and the eval verifier result. One model per
process so the GPU is released on exit.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adaptors.prompt_format import (  # noqa: E402
    c2_response_think_flags,
    render_c2_prompt,
)
from adaptors.verl_aligned_adaptor import _extract_boxed, _verify_math  # noqa: E402

PROBLEMS = (
    ("Compute 1+1.", "2"),
    ("What is 6*7?", "42"),
    ("Compute 12-5.", "7"),
    ("What is 3 cubed?", "27"),
)
STOPS = [151645, 151643]


def _run_one(model: str) -> int:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    prompts = []
    truths = []
    ids_per = []
    for problem, truth in PROBLEMS:
        text = render_c2_prompt(problem)
        ids = list(tok.encode(text, add_special_tokens=False))
        prompts.append(text)
        truths.append(truth)
        ids_per.append(ids)
    print(
        json.dumps(
            {
                "model": model,
                "prompt_format": "c2_nothink",
                "n_prompts": len(prompts),
                "prompt_ids": ids_per,
                "prompt_lens": [len(x) for x in ids_per],
                "stop_token_ids": STOPS,
                "temperature": 0.6,
                "top_p": 0.95,
                "max_tokens": 1024,
            }
        ),
        flush=True,
    )
    llm = LLM(
        model=model,
        tensor_parallel_size=1,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.5,
        max_num_seqs=4,
    )
    params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=1024,
        stop_token_ids=STOPS,
    )
    outputs = llm.generate(prompts, params)
    rows = []
    for (problem, truth), out in zip(PROBLEMS, outputs):
        comp = out.outputs[0]
        text = comp.text or ""
        gen_ids = [int(x) for x in (comp.token_ids or [])]
        extracted = _extract_boxed(text)
        try:
            verified = bool(extracted) and bool(_verify_math(extracted, truth))
        except Exception as exc:  # noqa: BLE001
            verified = False
            extracted = f"{extracted} (verify error: {exc})"
        flags = c2_response_think_flags(text)
        rows.append(
            {
                "problem": problem,
                "ground_truth": truth,
                "n_tokens": len(gen_ids),
                "terminal_token_id": gen_ids[-1] if gen_ids else None,
                "finish_reason": comp.finish_reason,
                "truncated": comp.finish_reason == "length" or len(gen_ids) >= 1024,
                "extracted": extracted,
                "verified": verified,
                **flags,
            }
        )
    print(json.dumps({"model": model, "samples": rows}, ensure_ascii=False), flush=True)
    del llm
    gc.collect()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", default=[])
    args = parser.parse_args()
    models = args.model or [
        "/root/autodl-tmp/models/Qwen/Qwen3-1.7B",
        "/root/autodl-tmp/models/Qwen/Qwen3-1.7B-Base",
        "/root/autodl-tmp/models/Qwen/Qwen3-4B",
    ]
    # Separate process per model so vLLM releases the GPU.
    if len(models) == 1:
        return _run_one(models[0])
    import subprocess

    rc = 0
    for model in models:
        print(f"=== smoke model {model} ===", flush=True)
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--model", model],
            check=False,
        )
        rc = max(rc, int(proc.returncode))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
