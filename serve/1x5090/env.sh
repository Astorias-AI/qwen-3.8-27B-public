#!/usr/bin/env bash
# Shared pins for Qwen3.8-27B NVFP4 on 1x RTX 5090 — throughput / prefill max.
# Profile: serve/1x5090/
# CUDA is provided by the Vast image — do not install a toolkit from this repo.

set -euo pipefail

# --- Models (local volume paths after download) ---
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

# --- 1x5090 throughput pins ---
# Official cookbook C=1 / chunk 1024 is the latency cell. C=16 / mamba-48
# does not fit NVFP4 + DFlash2 + GDN verify on 32GB with current SGLang.
#
# extra_buffer_lazy S=4, plus decode-lock skip → S=3.
# C=4 → max-mamba-cache-size = 4*3 ≈ 8 (slot pin overrides mamba ratio).
# HiCache (stage 3) spills KV to host so the GPU keeps state + graphs + prefill.
TP_SIZE="${TP_SIZE:-1}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flashinfer}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"
PAGE_SIZE="${PAGE_SIZE:-16}"
MAMBA_RADIX_STRATEGY="${MAMBA_RADIX_STRATEGY:-extra_buffer_lazy}"
MAMBA_SSM_DTYPE="${MAMBA_SSM_DTYPE:-bfloat16}"
SPEC_NUM_DRAFT_TOKENS="${SPEC_NUM_DRAFT_TOKENS:-4}"
SPEC_DRAFT_KV_DTYPE="${SPEC_DRAFT_KV_DTYPE:-fp8_e4m3}"

# 1x5090 32GB: target 20.1 GB + draft 3.7 GB leaves ~6.8 GB.
# C=8 / mamba 24 / draft 8 fails the KV check (GDN verify budget).
# C=4 / mamba 8 clamps max_running_requests to 2 (3 slots/req).
# C=4 / mamba 12 actually runs 4 reqs; SGLang then sizes KV to ~28k
# tokens even if MAX_TOTAL_TOKENS is 65536. Prefill graphs at bs=4
# fit with mem-frac 0.90. Do not raise mem-frac toward 0.97.
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-4}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-4}"
CUDA_GRAPH_MAX_BS_PREFILL="${CUDA_GRAPH_MAX_BS_PREFILL:-4}"
MAX_MAMBA_CACHE_SIZE="${MAX_MAMBA_CACHE_SIZE:-12}"
CHUNKED_PREFILL="${CHUNKED_PREFILL:-2048}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-8192}"
MEM_FRACTION="${MEM_FRACTION:-0.90}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-65536}"
DISABLE_PREFILL_CUDA_GRAPH="${DISABLE_PREFILL_CUDA_GRAPH:-0}"

HICACHE_RATIO="${HICACHE_RATIO:-2}"
HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_through}"
HICACHE_IO_BACKEND="${HICACHE_IO_BACKEND:-kernel}"
SCHEDULE_POLICY="${SCHEDULE_POLICY:-lpm}"
SCHEDULE_CONSERVATIVENESS="${SCHEDULE_CONSERVATIVENESS:-0.3}"

# One fewer GDN slot per request → more concurrency from the same state pool.
export SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK="${SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK:-1}"

# Unbuffered logs so laptop `./vast/logs.sh --follow` shows each shard / Load weight line.
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-0}"
export TQDM_DISABLE="${TQDM_DISABLE:-0}"

# If boot still errors "no GPU memory for the KV cache": drop C / draft / mamba.
# If graph capture OOMs: keep DISABLE_PREFILL_CUDA_GRAPH=1 and/or cut MAX_TOTAL_TOKENS.

TTFT_P95_SLO_MS="${TTFT_P95_SLO_MS:-1200}"

export HF_HOME
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_XET_NUM_CONCURRENT_RANGE_GETS="${HF_XET_NUM_CONCURRENT_RANGE_GETS:-16}"

mkdir -p "${MODEL_DIR}" "${DRAFT_DIR}" "${HF_HOME}" "${LOG_DIR}"
