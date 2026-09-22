#!/usr/bin/env bash
# Idempotent Hugging Face download onto the Vast volume.
# Fast path: hf_transfer + Xet high-performance range gets.
# Pins/paths come from serve/${SERVE_PROFILE}/env.sh (default: 1x5090).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVE_PROFILE="${SERVE_PROFILE:-1x5090}"
ENV_FILE="${SCRIPT_DIR}/${SERVE_PROFILE}/env.sh"
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Unknown SERVE_PROFILE=${SERVE_PROFILE} (expected serve/1x5090 or serve/2x5090)" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "${ENV_FILE}"

need_snapshot() {
  local dir="$1"
  [[ -f "${dir}/config.json" ]] || return 0
  local weights
  weights="$(find "${dir}" -maxdepth 1 \( -name '*.safetensors' -o -name '*.bin' \) | head -n 1 || true)"
  [[ -z "${weights}" ]]
}

install_hf_cli() {
  if command -v hf >/dev/null 2>&1 || command -v huggingface-cli >/dev/null 2>&1; then
    python3 -c "import hf_transfer" 2>/dev/null && return 0
  fi
  echo "Installing huggingface_hub[cli,hf_xet] + hf_transfer"
  python3 -m pip install -q -U "huggingface_hub[cli,hf_xet]" hf_transfer
}

hf_download() {
  local repo="$1"
  local dest="$2"
  mkdir -p "${dest}"
  echo "=== files in ${repo} ==="
  python3 - "${repo}" <<'PY'
import os, sys
from huggingface_hub import HfApi
repo = sys.argv[1]
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
files = HfApi(token=token).list_repo_files(repo)
weights = [f for f in files if f.endswith((".safetensors", ".bin", ".pt", ".gguf"))]
meta = [f for f in files if f.endswith((".json", ".py", ".txt", ".model", ".tiktoken"))]
print(f"{len(files)} files ({len(weights)} weight shards)")
for f in weights:
    print(f"  weight  {f}")
for f in meta[:20]:
    print(f"  meta    {f}")
if len(meta) > 20:
    print(f"  ... {len(meta) - 20} more meta files")
PY
  echo "=== downloading ${repo} → ${dest} (tqdm / hf_transfer per shard) ==="
  if command -v hf >/dev/null 2>&1; then
    hf download "${repo}" --local-dir "${dest}"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "${repo}" --local-dir "${dest}" --resume-download
  else
    python3 - "${repo}" "${dest}" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2], resume_download=True)
PY
  fi
  echo "=== ${dest} after download ==="
  ls -lh "${dest}" | sed -n '1,50p'
}

install_hf_cli

echo "SERVE_PROFILE=${SERVE_PROFILE} (tp=${TP_SIZE} C=${MAX_RUNNING_REQUESTS} mamba=${MAX_MAMBA_CACHE_SIZE})"

if need_snapshot "${MODEL_DIR}"; then
  echo "Downloading ${MODEL_ID} → ${MODEL_DIR}"
  hf_download "${MODEL_ID}" "${MODEL_DIR}"
else
  echo "Target weights already present at ${MODEL_DIR}"
fi

if need_snapshot "${DRAFT_DIR}"; then
  echo "Downloading ${DRAFT_ID} → ${DRAFT_DIR}"
  hf_download "${DRAFT_ID}" "${DRAFT_DIR}"
else
  echo "Draft weights already present at ${DRAFT_DIR}"
fi

echo "Downloads complete."
ls -lh "${MODEL_DIR}" "${DRAFT_DIR}" | sed -n '1,40p'
