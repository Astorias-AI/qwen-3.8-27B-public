#!/usr/bin/env bash
# Stage 3 — default: NVFP4 + DFlash2 + HiCache KV offload.
#
# HiCache is hierarchical KV cache: GPU pages write-through to host RAM
# (--hicache-ratio 2 = host pool 2x GPU mem, kernel I/O backend).
# That is the KV offload. GDN state stays on GPU (it has to).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
start_sglang_stage dflash2_hicache
