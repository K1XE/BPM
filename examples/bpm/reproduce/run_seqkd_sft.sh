#!/usr/bin/env bash
# SeqKD reference: offline SFT on teacher-generated trajectories.
# Build SFT_DATA_PATH first with examples/bpm/data/gen_teacher_trajectories.py,
# then examples/bpm/data/build_seqkd_sft_data.py.
set -euo pipefail

: "${STUDENT_MODEL_PATH:?path to Qwen3.5-2B HF checkpoint}"
: "${STUDENT_REF_LOAD:?student converted to Megatron torch-dist}"
: "${SFT_DATA_PATH:?SFT jsonl from build_seqkd_sft_data.py}"
: "${SAVE_ROOT:=./checkpoints/seqkd}"
: "${LR:=1e-5}"
: "${MIN_LR:=1e-6}"
: "${LR_WARMUP_FRACTION:=0.03}"
: "${NUM_EPOCH:=3}"
: "${ROLLOUT_BSZ:=256}"
: "${GLOBAL_BSZ:=256}"
: "${MICRO_BSZ:=1}"
: "${MAX_TOKENS_PER_GPU:=32768}"
: "${CP_SIZE:=1}"
: "${TP_SIZE:=1}"
: "${PP_SIZE:=1}"
: "${SAVE_INTERVAL:=200}"
: "${NUM_GPUS:=8}"
: "${STUDENT_MODEL_CONF:=scripts/models/qwen3.5-2B.sh}"

# must be empty, or Megatron resumes from it and ignores --ref-load
: "${LOAD_DIR:=${SAVE_ROOT}/load}"
mkdir -p "${LOAD_DIR}"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export BPM_GDN_NO_BOUNDARY_RESET=1
source "${STUDENT_MODEL_CONF}"

CMD=(
   python train.py
   "${MODEL_ARGS[@]}"
   --hf-checkpoint "${STUDENT_MODEL_PATH}"
   --ref-load "${STUDENT_REF_LOAD}"
   --load "${LOAD_DIR}"
   --save "${SAVE_ROOT}"
   --save-interval "${SAVE_INTERVAL}"
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout
   --prompt-data "${SFT_DATA_PATH}"
   --input-key messages
   # qwen3 would strip the think block from the SFT target
   --loss-mask-type distill_qwen_think
   --rollout-shuffle
   --num-epoch "${NUM_EPOCH}"
   --rollout-batch-size "${ROLLOUT_BSZ}"
   --global-batch-size "${GLOBAL_BSZ}"
   --loss-type sft_loss
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --debug-train-only
   --actor-num-nodes 1
   --actor-num-gpus-per-node "${NUM_GPUS}"
   --tensor-model-parallel-size "${TP_SIZE}"
   --pipeline-model-parallel-size "${PP_SIZE}"
   --context-parallel-size "${CP_SIZE}"
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --micro-batch-size "${MICRO_BSZ}"
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
   --sequence-parallel
   --optimizer adam
   --lr "${LR}"
   --lr-decay-style cosine
   --min-lr "${MIN_LR}"
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.95
   --lr-warmup-fraction "${LR_WARMUP_FRACTION}"
   --train-env-vars '{"BPM_GDN_NO_BOUNDARY_RESET":"1"}'
)

if [[ "${DRY_RUN:-0}" != "0" ]]; then
   printf '%q ' "${CMD[@]}"
   echo
   exit 0
fi

exec "${CMD[@]}"
