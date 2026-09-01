#!/usr/bin/env bash
# BPM ablation: filter broken code
set -euo pipefail
export BPM_FILTER_BROKEN_CODE=on
exec bash "$(dirname "$0")/run_p2_glm_z1_9b.sh"
