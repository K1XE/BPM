#!/usr/bin/env bash
# BPM ablation: no stop bridge
set -euo pipefail
export BPM_STOP_BRIDGE_MODE=skip
exec bash "$(dirname "$0")/run_p2_glm_z1_9b.sh"
