#!/usr/bin/env bash
# Shared pins for Qwen3.8-27B NVFP4 on 2x RTX 5090 (TP=2).
# Measured defaults: C=8 / mamba=96 / draft=6 /
# prefill-graph=16 / decode-graph=16 / hicache=1 / mem-frac=0.86.
# CUDA is provided by the Vast image — do not install a toolkit from this repo.

set -euo pipefail

# --- Models (local volume paths after download) — unchanged ---
DATA_ROOT="${DATA_ROOT:-/data}"
MODEL_ID="${MODEL_ID:-RadixArk/Qwen3.8-27B-NVFP4}"
DRAFT_ID="${DRAFT_ID:-z-lab/Qwen3.8-27B-DFlash2}"
MODEL_DIR="${MODEL_DIR:-${DATA_ROOT}/models/Qwen3.8-27B-NVFP4}"
DRAFT_DIR="${DRAFT_DIR:-${DATA_ROOT}/models/Qwen3.8-27B-DFlash2}"
HF_HOME="${HF_HOME:-${DATA_ROOT}/huggingface}"
LOG_DIR="${LOG_DIR:-${DATA_ROOT}/logs}"
PID_FILE="${PID_FILE:-${LOG_DIR}/sglang.pid}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/sglang.log}"

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-27b}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"

# --- 2x5090 pins ---
TP_SIZE="${TP_SIZE:-2}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flashinfer}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"
PAGE_SIZE="${PAGE_SIZE:-16}"
MAMBA_RADIX_STRATEGY="${MAMBA_RADIX_STRATEGY:-extra_buffer_lazy}"
MAMBA_SSM_DTYPE="${MAMBA_SSM_DTYPE:-bfloat16}"
MAMBA_FULL_MEMORY_RATIO="${MAMBA_FULL_MEMORY_RATIO:-0.9}"
SPEC_NUM_DRAFT_TOKENS="${SPEC_NUM_DRAFT_TOKENS:-6}"
SPEC_DRAFT_KV_DTYPE="${SPEC_DRAFT_KV_DTYPE:-fp8_e4m3}"

MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-8}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-16}"
CUDA_GRAPH_MAX_BS_PREFILL="${CUDA_GRAPH_MAX_BS_PREFILL:-16}"
# extra_buffer_lazy uses ~3 mamba slots/request; C must satisfy mamba >= 3*C or SGLang caps C.
# 96 slots is the measured setting at C=8. A 24-slot floor was slower.
MAX_MAMBA_CACHE_SIZE="${MAX_MAMBA_CACHE_SIZE:-96}"
CHUNKED_PREFILL="${CHUNKED_PREFILL:-2048}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-16384}"
MEM_FRACTION="${MEM_FRACTION:-0.86}"
# GLOBAL LOCK: model-max context. Never lower this pin. Measured KV pool is an outcome of other knobs.
MAX_TOTAL_TOKENS=262144
DISABLE_PREFILL_CUDA_GRAPH="${DISABLE_PREFILL_CUDA_GRAPH:-0}"

HICACHE_RATIO="${HICACHE_RATIO:-1}"
HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_through}"
HICACHE_IO_BACKEND="${HICACHE_IO_BACKEND:-kernel}"
SCHEDULE_POLICY="${SCHEDULE_POLICY:-lpm}"
SCHEDULE_CONSERVATIVENESS="${SCHEDULE_CONSERVATIVENESS:-0.3}"

export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-16}"
export SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK="${SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK:-1}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-0}"
export TQDM_DISABLE="${TQDM_DISABLE:-0}"

# If boot errors "no GPU memory for the KV cache": cut C / mamba / draft — not max-total-tokens.
# If prefill graph capture OOMs: cut CUDA_GRAPH_MAX_BS_PREFILL (try 8) before mem-frac.

TTFT_P95_SLO_MS="${TTFT_P95_SLO_MS:-1200}"

export HF_HOME
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_XET_NUM_CONCURRENT_RANGE_GETS="${HF_XET_NUM_CONCURRENT_RANGE_GETS:-16}"

mkdir -p "${MODEL_DIR}" "${DRAFT_DIR}" "${HF_HOME}" "${LOG_DIR}"
