#!/usr/bin/env bash

# Paper pair P1: Qwen3-32B -> Qwen3.5-2B
# See README.md

set -euo pipefail
export STUDENT_MODEL_PATH="${STUDENT_MODEL_PATH:?path to Qwen3.5-2B HF checkpoint}"
export STUDENT_REF_LOAD="${STUDENT_REF_LOAD:?student converted to Megatron torch-dist}"
export BPM_GDN_NO_BOUNDARY_RESET=1
export STUDENT_MODEL_CONF=scripts/models/qwen3.5-2B.sh
export BPM_TEACHER_MODEL_PATH="${BPM_TEACHER_MODEL_PATH:?path to Qwen3-32B HF checkpoint}"
export TEACHER_TP_SIZE="${TEACHER_TP_SIZE:-2}"
export TEACHER_DP_SIZE="${TEACHER_DP_SIZE:-4}"
export DATA_PATH="${DATA_PATH:?on-policy prompt mix (see README: data schema)}"
export BPM_BETA="${BPM_BETA:-0}"
case "${BPM_BETA}" in
  1|1.0) export BPM_RKL_LAMBDA="${BPM_RKL_LAMBDA:-0.1}" ;;  # the paper's RKL arms use skew 0.1
esac
exec bash "$(dirname "$0")/../run_bpm_distill.sh"
