#!/usr/bin/env bash
# PAIR selects the teacher-student pair.
set -euo pipefail
export BASELINE=gold_unmatched
: "${PAIR:=p2_glm_z1_9b}"
case "${PAIR}" in
  p1*) exec bash "$(dirname "$0")/../reproduce/run_p1_qwen3_32b.sh" ;;
  p2*) exec bash "$(dirname "$0")/../reproduce/run_p2_glm_z1_9b.sh" ;;
  p3*) exec bash "$(dirname "$0")/../reproduce/run_p3_minimax_m27.sh" ;;
  *) echo "unknown PAIR=${PAIR}" >&2; exit 2 ;;
esac
