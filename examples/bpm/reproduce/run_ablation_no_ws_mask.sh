#!/usr/bin/env bash
# BPM ablation: no ws mask
set -euo pipefail
export BPM_MASK_WS_ROWS=off
exec bash "$(dirname "$0")/run_p2_glm_z1_9b.sh"
