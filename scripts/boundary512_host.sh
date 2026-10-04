#!/usr/bin/env bash
# Sequential C.2 n=512 boundary eval. One model finishes (rc 0 and COMPLETE)
# before the next starts. Safe to re-run: a model with COMPLETE is skipped,
# and the runner resumes unfinished chunks.
#
# tmux:
#   tmux new -s bnd512 'OPD_PROMPT_FORMAT=c2_nothink bash scripts/boundary512_host.sh OUT name=HF ...'
#
#   OPD_PROMPT_FORMAT=c2_nothink bash scripts/boundary512_host.sh \
#     /root/autodl-tmp/data/processed/c2_boundary512_20261005 \
#     base=/root/autodl-tmp/models/Qwen3-1.7B-Base \
#     g2_s300=.../v7_nothink_C01_base/runs/20261002T234643Z_h5_c2g2pilot/checkpoints/global_step_300/hf \
#     g3_s300=.../v7_nothink_Ours_base/runs/20261003T112527Z_h5_c2g3pilot/checkpoints/global_step_300/hf
#
# Sampler is fixed inside the runner: T 0.6, top_p 0.95, 10240, seed 20261005,
# c2_nothink, stops 151645/151643, 5 GPUs. Do not export a different temperature.
#
# AMC file in this repo is 40 problems, not the pre-registered 50
# (missing 2023 AMC 12A #9 #15 #18 and 12B #15 #18 #19 #20 #21 #22 #25).
# The run records that gap and still evaluates the on-disk 90+40.
# Pass --strict-registered through EXTRA_ARGS to refuse to start.
set -euo pipefail

if [[ "${OPD_PROMPT_FORMAT:-}" != "c2_nothink" ]]; then
  echo "OPD_PROMPT_FORMAT must be c2_nothink" >&2
  exit 2
fi
export OPD_PROMPT_FORMAT
export OPD_PROMPT_FORMAT_REQUIRED=1

if [[ $# -lt 2 ]]; then
  echo "usage: $0 OUT_ROOT name=MODEL_DIR [name=MODEL_DIR ...]" >&2
  exit 2
fi

OUT_ROOT=$1
shift
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p "$OUT_ROOT/logs"
LOG="$OUT_ROOT/logs/boundary512_host.log"
N_GPUS="${N_GPUS:-5}"
CHUNK_PROBLEMS="${CHUNK_PROBLEMS:-10}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

{
  echo "HOST_START $(date -u +%Y-%m-%dT%H:%M:%SZ) out=$OUT_ROOT n_gpus=$N_GPUS"
  echo "NOTE amc23 on disk is 40, registered 50; this host script does not pass --strict-registered"
} | tee -a "$LOG"

for spec in "$@"; do
  name="${spec%%=*}"
  model="${spec#*=}"
  if [[ -z "$name" || -z "$model" || "$name" == "$spec" ]]; then
    echo "bad model spec: $spec" | tee -a "$LOG" >&2
    exit 2
  fi
  marker="$OUT_ROOT/$name/COMPLETE"
  if [[ -f "$marker" ]]; then
    echo "SKIP $name marker=$marker" | tee -a "$LOG"
    continue
  fi
  echo "MODEL_START $name $(date -u +%Y-%m-%dT%H:%M:%SZ) dir=$model" | tee -a "$LOG"
  set +e
  # shellcheck disable=SC2086
  python -m opd_eval.boundary512 \
    --model-dir "$model" \
    --out-dir "$OUT_ROOT/$name" \
    --sets aime24,aime25,aime26,amc23 \
    --n 512 \
    --seed 20261005 \
    --chunk-problems "$CHUNK_PROBLEMS" \
    --n-gpus "$N_GPUS" \
    $EXTRA_ARGS \
    2>&1 | tee -a "$LOG" "$OUT_ROOT/logs/${name}.log"
  rc=${PIPESTATUS[0]}
  set -e
  echo "EVAL_RC $rc model=$name" | tee -a "$LOG"
  if [[ "$rc" -ne 0 ]]; then
    exit "$rc"
  fi
  if [[ ! -f "$marker" ]]; then
    echo "missing completion marker $marker" | tee -a "$LOG" >&2
    exit 1
  fi
done

echo "HOST_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
