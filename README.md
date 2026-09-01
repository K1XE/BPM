<div align="center">

# 🧩 Cross-Tokenizer On-Policy Distillation<br>via Byte-Prefix Marginalization

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/2607.22334)
[![Model](https://img.shields.io/badge/Model-Hugging_Face-FFD21E.svg)](https://huggingface.co/K1zE/BPM)
[![Data](https://img.shields.io/badge/Data-mix--20k-2F80ED.svg)](https://huggingface.co/datasets/K1zE/BPM)
[![License](https://img.shields.io/badge/License-Apache--2.0-4C1.svg)](./LICENSE)

</div>

## 📣 Latest News

- **2026-09-01:** Released the [code](https://github.com/K1XE/BPM).
- **2026-07-27:** Released the [distilled checkpoint](https://huggingface.co/K1zE/BPM) and [`mix-20k` training data](https://huggingface.co/datasets/K1zE/BPM).
- **2026-07-24:** Released the [paper](https://arxiv.org/abs/2607.22334) and [project page](https://bpm-opd.github.io/).

<div align="center">
  <img width="1771" height="437" alt="BPM results overview" src="https://github.com/user-attachments/assets/78f01fdd-c420-4a24-b7d4-74b49a10c081" />
</div>

## 🔥 Overview

BPM distills a teacher into a student with a different tokenizer by marginalizing teacher probabilities over byte-prefix-compatible token paths.

- **Mass preserving:** no teacher probability is dropped at tokenizer boundaries.
- **On policy:** training follows the student's own rollouts.
- **Flexible objective:** forward KL, symmetric JSD, and reverse KL are supported.

<div align="center">
  <img width="2770" height="1334" alt="Byte-Prefix Marginalization" src="https://github.com/user-attachments/assets/3fa90ba8-748c-4a45-a29b-5a5fddc50b4f" />
</div>

## 📊 Results

<div align="center">
  <img width="1990" height="1200" alt="Table 1: main BPM results from the paper" src="https://github.com/user-attachments/assets/da230b9c-d7d3-47c2-a285-8277ce179191" />
</div>

## 🛠️ Setup

Follow the [slime environment setup](https://thudm.github.io/slime/get_started/quick_start.html), then install:

```bash
pip install "sglang==0.5.12.post1" math-verify pylatexenc

export MEGATRON_LM_PATH=/path/to/Megatron-LM
export PYTHONPATH="${MEGATRON_LM_PATH}:${PYTHONPATH}"
```

Convert the Qwen3.5-2B student checkpoint:

```bash
source scripts/models/qwen3.5-2B.sh

torchrun --nproc-per-node 1 tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint /path/to/Qwen3.5-2B/hf_checkpoint \
  --save /path/to/Qwen3.5-2B/megatron_torch_dist
```

## 🚀 Reproduce

Reference hardware: **1 × 8-H200 SXM node (141 GB/GPU)**.

Download [`mix-20k`](https://huggingface.co/datasets/K1zE/BPM), set the paths, and launch:

```bash
export STUDENT_MODEL_PATH=/path/to/Qwen3.5-2B/hf_checkpoint
export STUDENT_REF_LOAD=/path/to/Qwen3.5-2B/megatron_torch_dist
export BPM_TEACHER_MODEL_PATH=/path/to/GLM-Z1-9B-0414/hf_checkpoint
export DATA_PATH=/path/to/mix-20k.jsonl
export SAVE_ROOT=/path/to/checkpoints

bash examples/bpm/reproduce/run_p2_glm_z1_9b.sh
```

| Pair | Teacher | Launcher |
| --- | --- | --- |
| P1 | Qwen3-32B | `examples/bpm/reproduce/run_p1_qwen3_32b.sh` |
| P2 | GLM-Z1-9B | `examples/bpm/reproduce/run_p2_glm_z1_9b.sh` |
| P3 | MiniMax-M2.7 | `examples/bpm/reproduce/run_p3_minimax_m27.sh` |

<details>
<summary><b>Baselines and ablations</b></summary>

```bash
# simct | uld | gold | gold_matched | gold_unmatched
BASELINE=simct bash examples/baselines/reproduce/run_p2_glm_z1_9b.sh

# SeqKD
bash examples/bpm/reproduce/run_seqkd_sft.sh

# Ablations
bash examples/bpm/reproduce/run_ablation_no_ws_mask.sh
bash examples/bpm/reproduce/run_ablation_random_mask.sh
bash examples/bpm/reproduce/run_ablation_no_stop_bridge.sh
bash examples/bpm/reproduce/run_ablation_filter_broken_code.sh
```

</details>

## 🧪 Evaluation

Input JSONL: `{"prompt": [...], "label": ..., "metadata": {...}}`

```bash
python examples/bpm/eval/prepare_code_eval_data.py \
  --eval-dir /path/to/eval \
  --out-dir /path/to/eval

MODEL_PATH=/path/to/checkpoint/hf \
EVAL_SETS="math500:/path/to/eval/math500.jsonl humanevalplus:/path/to/eval/humanevalplus_codesandbox.jsonl" \
bash examples/bpm/eval/run_eval.sh
```

<details>
<summary><b>Benchmark sources</b></summary>

| Benchmark | Source |
| --- | --- |
| AIME 2026, HMMT 2026 | [MathArena](https://matharena.ai/) |
| MATH-500 | [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) |
| HumanEval+ | [EvalPlus](https://github.com/evalplus/evalplus) |
| LiveCodeBench | [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench), release_v6, 25.01-25.04 |
| TACO | [BAAI/TACO](https://huggingface.co/datasets/BAAI/TACO), official test split |

</details>

## ⭐ Citation

```bibtex
@article{DBLP:journals/corr/abs-2607-22334,
  author       = {Hao Wang and
                  Kun Yuan and
                  Wenlin Zhong and
                  Minglei Zhang and
                  Han Xiao and
                  Ming Sun and
                  Honggang Qi},
  title        = {Cross-Tokenizer On-Policy Distillation via Byte-Prefix Marginalization},
  journal      = {CoRR},
  volume       = {abs/2607.22334},
  year         = {2026},
  url          = {https://doi.org/10.48550/arXiv.2607.22334},
  doi          = {10.48550/ARXIV.2607.22334},
  eprinttype   = {arXiv},
  eprint       = {2607.22334},
  timestamp    = {Thu, 13 Aug 2026 08:18:11 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2607-22334.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

## 🔗 Related Projects

- [slime](https://github.com/THUDM/slime) — training framework.
- [KDFlow](https://github.com/songmzhang/KDFlow) — sleep/wake mechanism.

## 📄 License

[Apache 2.0](./LICENSE)
