# Training ↔ evaluation contract

This is the data contract. Training writes checkpoints; evaluation reads them and writes per-problem sample jsonl. Implementation of eval is not in this repo yet.

Invariant: **every evaluation run is `g=0`** (no teacher prefix, no hint). Same student chat template as training at `g=0` (proposal Appendix C.1 / C.3).

### The g=0 prompt is a cross-side invariant

Training at `g=0` and evaluation must render the **byte-identical** string. This is the golden form; both sides test against it, and neither side may change it unilaterally.

The authoritative machine-readable copy is `contract/g0_prompt.golden.json` (JSON-escaped, so a trailing newline cannot be silently normalized). This fenced block is documentation; `tests/test_c1_prompt.py` fails if the two diverge.

```
<|im_start|>system
You are an expert mathematician with strong problem-solving skills. Think step by step.<|im_end|>
<|im_start|>user
{problem}
Please reason step by step, and put your final answer within \boxed{}.<|im_end|>
<|im_start|>assistant
<think>

```

Conventions, frozen 2026-09-22:

- `add_generation_prompt=True` opens the assistant turn.
- A **format-only** `<think>\n` IS prefilled, carrying no teacher content. Appendix C.3's `g=0` row permits it, and it is required for continuity of the `g`-path: for `g≥1` the assistant side is a true token prefix of `T(x)`, which itself opens with `<think>\n` (Appendix C.1). Omitting it at `g=0` would put a discontinuity exactly on the endpoint that vanilla OPD (run C01) and the Base evaluations live on.
- `enable_thinking` is NOT passed. Passing `False` injects a *closed* empty think block, which is a different string.
- Students are `-Base` checkpoints, so the chat template must be verified against the real `Qwen/Qwen3-1.7B-Base` and `Qwen/Qwen3-0.6B-Base` tokenizers on the GPU host. Until then this golden string is **pending verification** and the tests say so rather than passing silently.

Verifier: Math-Verify on the student completion (boxed extract), same rule as corpus construction. The training verifier gate also looks only at the student's suffix; eval has no prefix to ignore.

---

## 1. Checkpoint path

Training writes under `train/examples/`. Canonical directory (eval loads **only** `hf/`):

```
train/examples/{example_id}/runs/{experiment_id}/checkpoints/global_step_{step}/
  hf/                 # HuggingFace-loadable (config.json, tokenizer, *.safetensors)
  manifest.json       # sidecar, required
```

The previous convention `train/outputs/{run_id}/{student_slug}/global_step_{step}/` is **superseded and illegal**. A path that still parses under it is refused — do not treat a leftover `outputs/` tree as a checkpoint.

| Piece | Rule |
| --- | --- |
| `example_id` | §5.1 table-row directory: `C01`, `C03`, `C04`, `C05`, `M1`, `S2`, `M2`, `L2`, `T1`, `F1`, `R06_C01`, `R06_C03`, `R06_M1`, `R06_S2`. |
| `experiment_id` | One launch of that config. `YYYYMMDDTHHMMSSZ` plus optional `_{hw}` / tag / `_s{seed}`. Also stored in the manifest. |
| `run_id` (manifest) | §5.1 method ID: `C01`, `C03`, `C04`, `C05`, `M1`, `S2`, `M2`, `L2`, `T1`, `F1`. R06 reuses those IDs and changes only `student_slug`; `example_id` is `R06_C01` etc. |
| `student_slug` | `qwen3-1.7b-base` or `qwen3-0.6b-base` |
| `step` | Integer global step, no zero-pad (`global_step_1000`). Full-step ckpts are step **1000** (lead 2026-09-21; proposal §5.1 says 1200 — do not “correct” back). F1 / D3 stay 200. |
| Format | Eval loads **only** `hf/` via `transformers`. FSDP / verl shards stay under `actor/` (or otherwise outside `hf/`) and are not part of this contract. |

`C00` / `C00p` (C00′) do not train. They evaluate the public Base snapshot (`Qwen/Qwen3-1.7B-Base` or `Qwen/Qwen3-0.6B-Base`) at two eval seeds. No `train/examples/C00/runs/…` checkpoint.

`manifest.json` (required at the `global_step_{step}/` directory):

```json
{
  "run_id": "C01",
  "example_id": "C01",
  "experiment_id": "20260922T035301Z_h4",
  "student_hf_id": "Qwen/Qwen3-1.7B-Base",
  "student_slug": "qwen3-1.7b-base",
  "teacher_hf_id": "Qwen/Qwen3-4B",
  "step": 1000,
  "g_eval": 0
}
```

`g_eval` must be `0`. Eval refuses the checkpoint if it is missing or nonzero. `run_id` and `experiment_id` are required so evaluation can map a checkpoint to the §5.1 table regardless of directory layout. `example_id` is required so the `examples/` folder is recoverable (needed for R06, where `run_id` is the method ID).

---

## 2. Sample-level jsonl

Canonical merge product — one file per `(run_id, student_slug, step, benchmark)`:

```
eval/outputs/{run_id}/{student_slug}/global_step_{step}/{benchmark_id}/samples.jsonl
```

Evaluation is embarrassingly parallel and **must** be shardable across processes, GPUs, and machines. A single process may write `samples.jsonl` directly (equivalent to one shard `000of001`). Otherwise each worker writes a shard; a merge step produces the canonical file. The merged result must be **identical** regardless of how the work was partitioned.

### Shard path and ownership

```
eval/outputs/{run_id}/{student_slug}/global_step_{step}/{benchmark_id}/
  samples.jsonl
  shards/
    {shard_idx:03d}of{shard_count:03d}.jsonl
    {shard_idx:03d}of{shard_count:03d}.manifest.json
```

Example: `shards/000of008.jsonl` + `shards/000of008.manifest.json`. `shard_idx` is 0-based. `shard_count` is the intended partition size, not “how many files exist yet”.

Each shard **declares** the `(problem_id, sample_idx)` slice it owns in the sidecar manifest. Both axes are allowed, so K=512 can be split into independent jobs (by `sample_idx`, by `problem_id`, or a rectangle) and a failed shard can be resumed without rerunning the others.

```json
{
  "shard_idx": 0,
  "shard_count": 8,
  "sample_idx_start": 0,
  "sample_idx_end": 64,
  "problem_ids": null
}
```

| Field | Rule |
| --- | --- |
| `sample_idx_start` / `sample_idx_end` | Half-open **global** range. The example owns `sample_idx` 0..63 on every problem. A problem-sharded job sets the range to `[0, K)` and lists `problem_ids`. |
| `problem_ids` | `null` = every problem in the benchmark. Otherwise an explicit list. Empty list is illegal. |
| Ownership | Pairwise disjoint across shards of the same `shard_count`. Union must cover the intended `(problem_id, sample_idx)` set (for AIME large-K: every problem × `{0,…,511}`). |

**`sample_idx` is global and must not be renumbered locally.** A shard that owns 64–127 writes `sample_idx: 64`, not `0`. avg@8 is defined as `sample_idx` 0–7 of the **same pooled file**. Per-problem $\hat p = (k+\tfrac12)/(K+1)$ uses the pooled count $K$, not the shard’s local length. A local 0..N-1 remap silently corrupts both.

Resume: re-run only incomplete shards with the **same** manifest. Within a shard file, skip `(problem_id, sample_idx)` pairs already present; do not append a second row for the same key.

### Merge rule

1. Read every `shards/{idx:03d}of{n:03d}.jsonl` for a fixed `n = shard_count`.
2. Union the records. Dedupe key is `(problem_id, sample_idx)`.
3. If two shards emit the same key: identical `verified` / `response` / `n_tokens` → keep one; any disagreement → **fail the merge**.
4. Completeness: after merge, every problem has exactly `sample_idx ∈ {0,…,K-1}` (no gaps, no extras).
5. Write `samples.jsonl` sorted by `(problem_id, sample_idx)`. Any partition of the same ownership covering the same keys must yield a byte-identical file after that sort.

Downstream metrics read **only** `samples.jsonl`. They must not depend on shard filenames or `shard_count`.

One record per sampled completion. Required fields:

| Field | Type | Semantics | Downstream |
| --- | --- | --- | --- |
| `problem_id` | string | Stable id, unique within the benchmark. Encoding is **not** fixed in §5.1 (lead must freeze the map). | Pass@K, learned/forgotten vs Base, per-problem $\Delta\log\hat p$, two-level bootstrap (resample problems), McNemar pairing. |
| `sample_idx` | int | `0 .. n_samples-1` for that problem. | Pass@K / avg@8 (need $K$ or 8 distinct idx); bootstrap layer 2 (resample completions). |
| `response` | string | Full student completion at `g=0` (includes think / boxed if produced). | Verifier input; length / diversity diagnostics if recomputed. |
| `verified` | bool | Math-Verify accept/reject on `response`. | $k=\sum$ `verified`; pass@K $=1[k\ge 1]$; avg@8 $=\overline{\texttt{verified}}$ over 8 samples; $\hat p=(k+\tfrac12)/(K+1)$; McNemar on the pass@K bit. |
| `n_tokens` | int | Completion token count (student suffix only). | Length / waste diagnostics. Not used for pass@K. |

Example:

```json
{"problem_id":"aime24:2024-i-1","sample_idx":0,"response":"<think>\n...\\n</think>\n\\boxed{123}","verified":true,"n_tokens":1420}
```

`problem_id` in the example is illustrative. §5.1 does not specify the id scheme.

Recommended extras (not required): `benchmark_id`, `run_id`, `step`, `eval_seed`, `temperature`, `g` (must be 0).

### How fields feed the §5.1 statistics

Let $K$ be the protocol $K$ for that surface (below). Per problem, $k=\#\{\text{verified}=\text{true}\}$.

- **pass@K**: $1[k\ge 1]$ (unbiased estimator may be used later; the contract only guarantees raw bits).
- **avg@8**: mean of `verified` over `sample_idx` 0–7 on the AIME union.
- **learned / forgotten vs Base**: compare pass@K bits. learned = Base 0 → ckpt 1; forgotten = Base 1 → ckpt 0. On OR1-200, each count is divided by the feasible denominator (learned / # Base-unsolvable in the bin; forgotten / # Base-solvable in the bin).
- **$\Delta\log\hat p$**: $\hat p=(k+\tfrac12)/(K+1)$ (shrinkage); $\Delta$ is ckpt minus Base on the same `problem_id`.
- **Two-level bootstrap**: resample problems × resample the $K$ completions.
- **McNemar**: paired pass@K bits vs Base on the same problems.

`C00` vs `C00p` (two Base eval seeds) is the noise floor: $N_K(\mathrm{Base},\mathrm{Base}')$ and $\Delta\log\hat p(\mathrm{Base},\mathrm{Base}')$.

Training-only curves $M_0$, $M_{\rm tf}$, and the terminal $g_{\rm curr}/m$ histogram are **not** in this jsonl.

---

## 3. Benchmark roster (from proposal §5.1)

| Surface | What §5.1 names | What eval must produce |
| --- | --- | --- |
| AIME union | AIME24+25+26 并集（90 题） | pass@$K$; learned/forgotten vs Base; **AIME26 as its own row** |
| AIME union | 同并集 avg@8 | avg@8 |
| Nine-benchmark suite | 「九基准」only — **names not listed** | pass@1, 8-sample average, **6 ckpts only** |
| Harm / OOD | SciBench; GPQA-D; 「一个零训练 coding 基准」(**unnamed**) | hung on the **same 6 ckpts**; 「有没有伤」only |
| Held-out bins | OR1-200, same bin protocol as the corpus | $K=128$: $\Delta\log\hat p$, learned/forgotten / feasible denoms, Base absolute level |

OR1-200 bins (corpus protocol, applied to held-out): **B0** $=0/16$; **B1–B3** = equal-mass tertiles of the rest; paper reports the realized cuts. Bins are a property of Base pass@16, not of the jsonl schema.

§5.1 does **not** name the nine benchmarks, the coding set, the 6 ckpt IDs, or a `problem_id` map. The §5.5 budget line lists nine-benchmark ckpts as C00, C01, C04, M1, S2, T1 — that list is **not** in §5.1.

`benchmark_id` strings to use once the lead names the missing sets (placeholders, not official names): `aime_union`, `aime26`, `or1_200`, plus one id per named/TBD set.

---

## 4. $K$ and temperatures (from §5.1 only)

§5.1 states $T=1$ only for the nine-benchmark pass@1 and the corpus pass@16. The lead froze the rest on 2026-09-21: **$T=1$ on every surface**, and **avg@8 is `sample_idx` 0–7 of the same pass@$K$ pool**, not a separate draw (so avg@8 and pass@$K$ are paired on identical completions). See `docs/eng/decisions.md`.

| Surface | $K$ / samples | Temperature |
| --- | --- | --- |
| AIME union pass@$K$ | **512** (lead 2026-09-21; Appendix E Q4 closed, floor kept). Sensitivity set $\{64,128,256,512\}$. | $T=1$ (lead, 2026-09-21) |
| AIME union avg@8 | `sample_idx` 0–7 of the pass@$K$ pool (shared draw, lead 2026-09-21) | $T=1$ |
| Nine-benchmark pass@1 | 8 samples, then average | **$T=1$** (§5.1) |
| OR1-200 held-out | **$K=128$** | $T=1$ (lead, 2026-09-21) |
| SciBench / GPQA-D / coding | $K$ still **not stated** in §5.1 | $T=1$ (lead, 2026-09-21) |
| Corpus / binning pass@16 (not a trained-ckpt eval) | 16, $T=1$, `max_new_tokens=8192` | $T=1$ (listed so bins stay aligned; not an eval-of-ckpt protocol) |

Eval sampling is always `g=0`.
