#!/usr/bin/env bash

# Benchmark evaluation: rollout-only sampling + graders
# See README.md

set -euo pipefail

: "${MODEL_PATH:?HF checkpoint to evaluate}"
: "${EVAL_SETS:?space-separated name:path pairs of benchmark jsonl files}"
: "${STUDENT_MODEL_CONF:=scripts/models/qwen3.5-2B.sh}"
: "${NUM_GPUS:=8}"
: "${N_SAMPLES:=8}"
: "${EVAL_TEMPERATURE:=0.6}"
: "${EVAL_TOP_P:=0.95}"
: "${EVAL_TOP_K:=20}"
: "${EVAL_MAX_RESP_LEN:=27648}"
# use 0.70 for the code benchmarks
: "${SGLANG_MEM_FRACTION:=0.85}"

source "${STUDENT_MODEL_CONF}"

EVAL_PAIRS=()
for pair in ${EVAL_SETS}; do
   EVAL_PAIRS+=("${pair%%:*}" "${pair#*:}")
done

ROLLOUT_ARGS=(
   --hf-checkpoint "${MODEL_PATH}"
   --debug-rollout-only
   --prompt-data "${EVAL_SETS##*:}"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --apply-chat-template-kwargs "{\"enable_thinking\": true}"
   --rm-type dapo
   --reward-key acc
   --eval-reward-key acc
   --custom-rm-path examples.bpm.eval.eval_reward.reward_func
   --num-rollout 0
   --rollout-batch-size 8
   --n-samples-per-prompt 1
   --global-batch-size 8
   --rollout-max-response-len "${EVAL_MAX_RESP_LEN}"
   --rollout-skip-special-tokens
   --log-passrate
)

EVAL_ARGS=(
   --eval-interval 1
   --eval-prompt-data "${EVAL_PAIRS[@]}"
   --eval-input-key prompt
   --eval-label-key label
   --n-samples-per-eval-prompt "${N_SAMPLES}"
   --eval-max-response-len "${EVAL_MAX_RESP_LEN}"
   --eval-temperature "${EVAL_TEMPERATURE}"
   --eval-top-p "${EVAL_TOP_P}"
   --eval-top-k "${EVAL_TOP_K}"
)

PERF_ARGS=(
   --actor-num-nodes 1
   --actor-num-gpus-per-node "${NUM_GPUS}"
   --rollout-num-gpus "${NUM_GPUS}"
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION}"
)

CMD=(python train.py "${MODEL_ARGS[@]}" "${ROLLOUT_ARGS[@]}" "${PERF_ARGS[@]}" "${EVAL_ARGS[@]}")

if [[ "${DRY_RUN:-0}" != "0" ]]; then
   printf '%q ' "${CMD[@]}"
   echo
   exit 0
fi

exec "${CMD[@]}"
