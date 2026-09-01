#!/usr/bin/env bash
# BPM ablation: random mask
set -euo pipefail
export BPM_MASK_WS_ROWS=off
export BPM_MASK_RANDOM_ROWS=1
exec bash "$(dirname "$0")/run_p2_glm_z1_9b.sh"
