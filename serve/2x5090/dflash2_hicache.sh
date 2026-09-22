#!/usr/bin/env bash
# Stage 3 — 2x5090 default: NVFP4 + DFlash2 + HiCache KV offload.
#
# HiCache is hierarchical KV cache: GPU pages write-through to host RAM.
# The 2× recipe sets hicache-ratio to 1 (host pool matches GPU KV).
# GDN state stays on GPU.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
start_sglang_stage dflash2_hicache
