#!/usr/bin/env bash
# Eval under an explicit prompt format. Refuses an unset OPD_PROMPT_FORMAT.
# C.2: OPD_PROMPT_FORMAT=c2_nothink bash scripts/nothink_eval.sh mini [args...]
# or:  ... nothink_eval.sh full [args...]
set -euo pipefail
if [[ -z "${OPD_PROMPT_FORMAT:-}" ]]; then
  echo "OPD_PROMPT_FORMAT must be set to c1_think or c2_nothink" >&2
  exit 1
fi
case "$OPD_PROMPT_FORMAT" in
  c1_think|c2_nothink) ;;
  *)
    echo "OPD_PROMPT_FORMAT=$OPD_PROMPT_FORMAT is invalid" >&2
    exit 1
    ;;
esac
export OPD_PROMPT_FORMAT
export OPD_PROMPT_FORMAT_REQUIRED=1
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
kind="${1:-}"
if [[ "$kind" != "mini" && "$kind" != "full" ]]; then
  echo "usage: $0 mini|full [protocol args...]" >&2
  exit 2
fi
shift
if [[ "$kind" == "mini" ]]; then
  exec python -m opd_eval.mini_protocol --require-prompt-format --prompt-format "$OPD_PROMPT_FORMAT" "$@"
fi
exec python -m opd_eval.full_protocol --require-prompt-format --prompt-format "$OPD_PROMPT_FORMAT" "$@"
