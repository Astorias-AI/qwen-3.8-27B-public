#!/usr/bin/env bash
# Helpers for starting / stopping SGLang on the Vast instance.
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

sglang_cmd() {
  if command -v sglang >/dev/null 2>&1; then
    echo "sglang serve"
    return
  fi
  echo "python3 -m sglang.launch_server"
}

stop_sglang() {
  if [[ -f "${PID_FILE}" ]]; then
    local pid
    pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      echo "Stopping SGLang pid ${pid}"
      kill "${pid}" 2>/dev/null || true
      for _ in $(seq 1 30); do
        kill -0 "${pid}" 2>/dev/null || break
        sleep 1
      done
      kill -9 "${pid}" 2>/dev/null || true
    fi
    rm -f "${PID_FILE}"
  fi
  pkill -f "sglang.launch_server|sglang serve" 2>/dev/null || true
  sleep 1
}

# Build the argv for a stage into the global array SGLANG_ARGS.
build_sglang_args() {
  local stage="$1"
  case "${stage}" in
    baseline|dflash2|dflash2_hicache) ;;
    *)
      echo "unknown stage: ${stage}" >&2
      return 1
      ;;
  esac

  SGLANG_ARGS=(
    --model-path "${MODEL_DIR}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --trust-remote-code
    --tp-size "${TP_SIZE}"
    --kv-cache-dtype "${KV_CACHE_DTYPE}"
    --mem-fraction-static "${MEM_FRACTION}"
    --attention-backend "${ATTENTION_BACKEND}"
    --chunked-prefill-size "${CHUNKED_PREFILL}"
    --max-prefill-tokens "${MAX_PREFILL_TOKENS}"
    --page-size "${PAGE_SIZE}"
    --mamba-radix-cache-strategy "${MAMBA_RADIX_STRATEGY}"
    --max-mamba-cache-size "${MAX_MAMBA_CACHE_SIZE}"
    --mamba-ssm-dtype "${MAMBA_SSM_DTYPE}"
    --mamba-full-memory-ratio "${MAMBA_FULL_MEMORY_RATIO}"
    --max-running-requests "${MAX_RUNNING_REQUESTS}"
    --cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS}"
    --cuda-graph-max-bs-prefill "${CUDA_GRAPH_MAX_BS_PREFILL}"
    --schedule-policy "${SCHEDULE_POLICY}"
    --schedule-conservativeness "${SCHEDULE_CONSERVATIVENESS}"
    --reasoning-parser qwen3
    --tool-call-parser qwen3_coder
    --sampling-defaults model
    --enable-metrics
    --enable-cache-report
    --sleep-on-idle
    --host "${HOST}"
    --port "${PORT}"
  )

  if [[ -n "${MAX_TOTAL_TOKENS:-}" ]]; then
    SGLANG_ARGS+=(--max-total-tokens "${MAX_TOTAL_TOKENS}")
  fi
  if [[ "${DISABLE_PREFILL_CUDA_GRAPH:-0}" == "1" ]]; then
    SGLANG_ARGS+=(--disable-prefill-cuda-graph)
  fi

  if [[ "${stage}" == dflash2 || "${stage}" == dflash2_hicache ]]; then
    SGLANG_ARGS+=(
      --speculative-algorithm DFLASH
      --speculative-draft-model-path "${DRAFT_DIR}"
      --speculative-num-draft-tokens "${SPEC_NUM_DRAFT_TOKENS}"
      --speculative-draft-kv-cache-dtype "${SPEC_DRAFT_KV_DTYPE}"
    )
  fi

  # Hierarchical KV: GPU → host RAM (and optional disk). This is the offload.
  if [[ "${stage}" == dflash2_hicache ]]; then
    SGLANG_ARGS+=(
      --enable-hierarchical-cache
      --hicache-ratio "${HICACHE_RATIO}"
      --hicache-write-policy "${HICACHE_WRITE_POLICY}"
      --hicache-io-backend "${HICACHE_IO_BACKEND}"
    )
    if [[ -n "${HICACHE_SIZE_GB:-}" ]]; then
      SGLANG_ARGS+=(--hicache-size "${HICACHE_SIZE_GB}")
    fi
  fi

  if [[ -n "${SGLANG_API_KEY:-}" ]]; then
    SGLANG_ARGS+=(--api-key "${SGLANG_API_KEY}")
  fi
}

start_sglang_stage() {
  local stage="$1"
  build_sglang_args "${stage}"
  stop_sglang
  mkdir -p "${LOG_DIR}"

  local cmd
  cmd="$(sglang_cmd)"

  echo "=== Starting SGLang stage=${stage} ===" | tee -a "${LOG_FILE}"
  echo "NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS:-unset} MAMBA_FULL_MEMORY_RATIO=${MAMBA_FULL_MEMORY_RATIO} draft=${SPEC_NUM_DRAFT_TOKENS}" | tee -a "${LOG_FILE}"
  echo "cmd: ${cmd} ${SGLANG_ARGS[*]}" | tee -a "${LOG_FILE}"
  echo "Throughput envelope: C=${MAX_RUNNING_REQUESTS} graph-bs=${CUDA_GRAPH_MAX_BS} prefill-graph-bs=${CUDA_GRAPH_MAX_BS_PREFILL} chunk=${CHUNKED_PREFILL} mamba-slots=${MAX_MAMBA_CACHE_SIZE} mem-frac=${MEM_FRACTION} max-total-tokens=${MAX_TOTAL_TOKENS:-auto} disable-prefill-graph=${DISABLE_PREFILL_CUDA_GRAPH:-0}" | tee -a "${LOG_FILE}"
  echo "Check boot log for max_running_requests — it must not clamp below ${MAX_RUNNING_REQUESTS}." | tee -a "${LOG_FILE}"

  if command -v stdbuf >/dev/null 2>&1; then
    # shellcheck disable=SC2086
    nohup env NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS}" stdbuf -oL -eL ${cmd} "${SGLANG_ARGS[@]}" >>"${LOG_FILE}" 2>&1 &
  else
    # shellcheck disable=SC2086
    nohup env NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS}" ${cmd} "${SGLANG_ARGS[@]}" >>"${LOG_FILE}" 2>&1 &
  fi
  echo $! >"${PID_FILE}"
  echo "SGLang pid $(cat "${PID_FILE}") logging to ${LOG_FILE}"
}
