#!/usr/bin/env bash
# One-GPU C.2 smoke. Leaves the GPU free when the python process exits.
set -euo pipefail
source /root/autodl-tmp/opd-env.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec python "$ROOT/scripts/c2_vllm_smoke.py" "$@"
