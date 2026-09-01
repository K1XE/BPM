#!/usr/bin/env bash
# Paper pair P3: MiniMax-M2.7 -> Qwen3.5-2B, cross-tokenizer baselines
# BASELINE=simct|uld|gold|gold_matched|gold_unmatched
set -euo pipefail
export STUDENT_MODEL_PATH="${STUDENT_MODEL_PATH:?path to Qwen3.5-2B HF checkpoint}"
export STUDENT_REF_LOAD="${STUDENT_REF_LOAD:?student converted to Megatron torch-dist}"
export STUDENT_MODEL_CONF=scripts/models/qwen3.5-2B.sh
export TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-${BPM_TEACHER_MODEL_PATH:?path to MiniMax-M2.7 HF checkpoint}}"
export TEACHER_TP_SIZE="${TEACHER_TP_SIZE:-8}"
export TEACHER_DP_SIZE="${TEACHER_DP_SIZE:-1}"
export TEACHER_EP_SIZE="${TEACHER_EP_SIZE:-8}"
export DATA_PATH="${DATA_PATH:?on-policy prompt mix (see README: data schema)}"
exec bash "$(dirname "$0")/../run_baseline_distill.sh"
