#!/usr/bin/env bash

# Single-node cross-tokenizer distillation baseline.
# BASELINE selects the arm; see README.md. DRY_RUN=1 prints the command.

set -euo pipefail

: "${BASELINE:?set BASELINE to one of simct|uld|gold|gold_matched|gold_unmatched}"

case "${BASELINE}" in
  simct)
    BASELINE_BACKEND=simct
    BASELINE_ENTRY=slime_plugins.baselines.simct.entry.custom_loss.simct_custom_loss_function
    : "${OPD_LOSS_REDUCTION:=per_token}"
    : "${MAX_TOKENS_PER_GPU:=10368}"
    ;;
  uld|gold|gold_matched|gold_unmatched)
    BASELINE_BACKEND=gold
    BASELINE_ENTRY=slime_plugins.baselines.gold.entry.custom_loss.gold_custom_loss_function
    : "${OPD_LOSS_REDUCTION:=per_sample}"
    : "${MAX_TOKENS_PER_GPU:=6912}"
    ;;
  *)
    echo "unknown BASELINE=${BASELINE}" >&2; exit 2 ;;
esac

: "${BPM_GDN_NO_BOUNDARY_RESET:=1}"
export BPM_GDN_NO_BOUNDARY_RESET
TRAIN_ENV_JSON=$(printf '{"BPM_GDN_NO_BOUNDARY_RESET":"%s"}' "${BPM_GDN_NO_BOUNDARY_RESET}")

: "${STUDENT_MODEL_PATH:=./models/student}"
: "${STUDENT_REF_LOAD:=./models/student_ref}"
: "${TEACHER_MODEL_PATH:=${BPM_TEACHER_MODEL_PATH:-./models/teacher}}"
: "${TEACHER_TOKENIZER_PATH:=${TEACHER_MODEL_PATH}}"
: "${DATA_PATH:=./data/prompts.jsonl}"
: "${SAVE_ROOT:=./checkpoints}"
: "${DATA_INPUT_KEY:=prompt}"
: "${DATA_LABEL_KEY:=label}"
: "${CHAT_TEMPLATE_KWARGS:={\"enable_thinking\": true\}}"
: "${LR:=5e-7}"
: "${RM_TYPE:=dapo}"
: "${CUSTOM_RM_PATH:=examples.bpm.reward.reward_adapter.reward_func}"
: "${CUSTOM_REWARD_POST_PROCESS_PATH:=examples.bpm.reward.reward_adapter.post_process_rewards}"
: "${NUM_ROLLOUT:=156}"
: "${ROLLOUT_BSZ:=256}"
: "${GLOBAL_BSZ:=256}"
: "${N_SAMPLES_PER_PROMPT:=1}"
: "${MAX_RESP_LEN:=27648}"
: "${ROLLOUT_TEMPERATURE:=1.0}"
: "${ROLLOUT_TOP_P:=0.95}"
: "${ROLLOUT_TOP_K:=20}"
: "${SAVE_INTERVAL:=25}"

: "${EVAL_SETS:=}"
: "${EVAL_INTERVAL:=10}"
: "${EVAL_N_SAMPLES:=8}"
: "${EVAL_TEMPERATURE:=0.6}"
: "${EVAL_TOP_P:=0.95}"
: "${EVAL_TOP_K:=20}"
: "${EVAL_MAX_RESP_LEN:=${MAX_RESP_LEN}}"

: "${WANDB_PROJECT:=baseline-distill}"
: "${WANDB_API_KEY:=}"
: "${SGLANG_SERVED_MODEL_NAME:=student_model}"
: "${NUM_GPUS:=8}"
: "${TEACHER_TP_SIZE:=4}"
: "${TEACHER_DP_SIZE:=1}"
: "${TEACHER_EP_SIZE:=1}"
: "${TEACHER_PLACEMENT:=}"
: "${ROLLOUT_TP_SIZE:=1}"
: "${CP_SIZE:=4}"
: "${STUDENT_MODEL_CONF:=scripts/models/qwen3.5-2B.sh}"

STUDENT_TP_SIZE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

source "${STUDENT_MODEL_CONF}"

CKPT_ARGS=(
   --hf-checkpoint "${STUDENT_MODEL_PATH}"
   --ref-load "${STUDENT_REF_LOAD}"
   --load "${SAVE_ROOT}"
   --save "${SAVE_ROOT}"
   --save-interval "${SAVE_INTERVAL}"
)

ROLLOUT_ARGS=(
   --prompt-data "${DATA_PATH}"
   --input-key "${DATA_INPUT_KEY}"
   --label-key "${DATA_LABEL_KEY}"
   --apply-chat-template
   --apply-chat-template-kwargs "${CHAT_TEMPLATE_KWARGS}"
   --rollout-shuffle
   --rm-type "${RM_TYPE}"
   --reward-key acc
   --log-passrate
   --eval-reward-key acc
   --custom-rm-path "${CUSTOM_RM_PATH}"
   --custom-reward-post-process-path "${CUSTOM_REWARD_POST_PROCESS_PATH}"
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BSZ}"
   --global-batch-size "${GLOBAL_BSZ}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --rollout-max-response-len "${MAX_RESP_LEN}"
   --rollout-temperature "${ROLLOUT_TEMPERATURE}"
   --rollout-top-p "${ROLLOUT_TOP_P}"
   --rollout-top-k "${ROLLOUT_TOP_K}"
   --rollout-skip-special-tokens
)

EVAL_ARGS=()
if [[ -n "${EVAL_SETS}" ]]; then
   EVAL_PAIRS=()
   for pair in ${EVAL_SETS}; do
      EVAL_PAIRS+=("${pair%%:*}" "${pair#*:}")
   done
   EVAL_ARGS=(
      --eval-interval "${EVAL_INTERVAL}"
      --eval-prompt-data "${EVAL_PAIRS[@]}"
      --eval-input-key "${DATA_INPUT_KEY}"
      --eval-label-key "${DATA_LABEL_KEY}"
      --n-samples-per-eval-prompt "${EVAL_N_SAMPLES}"
      --eval-max-response-len "${EVAL_MAX_RESP_LEN}"
      --eval-temperature "${EVAL_TEMPERATURE}"
      --eval-top-p "${EVAL_TOP_P}"
      --eval-top-k "${EVAL_TOP_K}"
   )
fi

OPTIMIZER_ARGS=(
   --lr "${LR}"
   --lr-decay-style constant
   --lr-warmup-fraction 0.0
   --optimizer adam
   --adam-beta1 0.9
   --adam-beta2 0.999
   --weight-decay 0.01
)

PERF_ARGS=(
   --tensor-model-parallel-size "${STUDENT_TP_SIZE}"
   --context-parallel-size "${CP_SIZE}"
   --sequence-parallel
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
   --log-probs-max-tokens-per-gpu 32768
   --balance-data
   --actor-num-nodes 1
   --actor-num-gpus-per-node "${NUM_GPUS}"
   --rollout-num-gpus "${NUM_GPUS}"
   --rollout-num-gpus-per-engine "${ROLLOUT_TP_SIZE}"
   --colocate
   --offload-rollout
   --offload-train
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --attention-softmax-in-fp32
   --accumulate-allreduce-grads-in-fp32
   --attention-backend flash
   --no-gradient-accumulation-fusion
)

SGLANG_ARGS=(
   --sglang-context-length 40960
   --sglang-mem-fraction-static 0.85
)

OPD_ARGS=(
   --loss-type custom_loss
   --custom-loss-function-path "${BASELINE_ENTRY}"
   --opd-mode standalone_loss
   --opd-backend "${BASELINE_BACKEND}"
   --opd-loss-reduction "${OPD_LOSS_REDUCTION}"
   --opd-teacher-model-path "${TEACHER_MODEL_PATH}"
   --opd-teacher-tokenizer-path "${TEACHER_TOKENIZER_PATH}"
   --opd-teacher-tp-size "${TEACHER_TP_SIZE}"
   --opd-teacher-dp-size "${TEACHER_DP_SIZE}"
   --opd-teacher-ep-size "${TEACHER_EP_SIZE}"
)

METHOD_ARGS=()
case "${BASELINE}" in
  simct)
    METHOD_ARGS=(
       --simct-alignment-mode span
       --simct-span-ctkd-norm
       --opd-loss-type rkl
       --opd-temperature 1.0
       --opd-ce-weight 0.0
       --opd-topk 0
       --simct-chunk-size "${MAX_TOKENS_PER_GPU}"
       --simct-compile-bucket-size 1024
       --simct-overlap-chunk-size 0
       --simct-train-log-interval 16
    )
    ;;
  *)
    # the reference GOLD arms skip the advantage/log-prob prepass; the loss reads neither
    METHOD_ARGS=(
       --disable-compute-advantages-and-returns
       --gold-trl-faithful
       --gold-uld-token-merge-strategy observed
       --gold-distillation-weight 1.0
       --gold-ce-weight 0.0
       --gold-student-temperature 1.0
       --gold-teacher-temperature 1.0
       --gold-use-extended-uld
       --gold-skip-student-eos
       --gold-skip-teacher-eos
       --gold-chunk-size 32
    )
    if [[ "${BASELINE}" != "uld" ]]; then
       METHOD_ARGS+=(--gold-use-hybrid-loss --gold-beta 0.5)
    fi
    case "${BASELINE}" in
      gold_matched)   METHOD_ARGS+=(--gold-hybrid-matched-weight 1.0 --gold-hybrid-unmatched-weight 0.0) ;;
      gold_unmatched) METHOD_ARGS+=(--gold-hybrid-matched-weight 0.0 --gold-hybrid-unmatched-weight 1.0) ;;
    esac
    ;;
esac

PLACEMENT_ARGS=()
if [[ -n "${TEACHER_PLACEMENT}" ]]; then
   PLACEMENT_ARGS=(--opd-teacher-placement "${TEACHER_PLACEMENT}")
fi

TRAIN_ENV_ARGS=(--train-env-vars "${TRAIN_ENV_JSON}")

export WANDB_PROJECT WANDB_API_KEY SGLANG_SERVED_MODEL_NAME

CMD=(
   python train.py
   "${MODEL_ARGS[@]}"
   "${CKPT_ARGS[@]}"
   "${ROLLOUT_ARGS[@]}"
   "${OPTIMIZER_ARGS[@]}"
   "${PERF_ARGS[@]}"
   "${MISC_ARGS[@]}"
   "${SGLANG_ARGS[@]}"
   "${OPD_ARGS[@]}"
   "${METHOD_ARGS[@]}"
   "${EVAL_ARGS[@]}"
   "${PLACEMENT_ARGS[@]}"
   "${TRAIN_ENV_ARGS[@]}"
)

if [[ "${DRY_RUN:-0}" != "0" ]]; then
   printf '%q ' "${CMD[@]}"
   echo
   exit 0
fi

exec "${CMD[@]}"
