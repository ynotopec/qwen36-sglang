#!/usr/bin/env bash

# Usage:
#   source ./run.sh [IP] [PORT]
# or:
#   ./run.sh [IP] [PORT]

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="$(basename "${PROJECT_DIR}")"

if [[ -f "${PROJECT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${PROJECT_DIR}/.env"
  set +a
fi

load_example_defaults() {
  local example_file="${PROJECT_DIR}/.env.example"
  [[ -f "${example_file}" ]] || return 0

  while IFS= read -r line; do
    [[ "${line}" =~ ^#[A-Z0-9_]+= ]] || continue

    local kv="${line#\#}"
    local var_name="${kv%%=*}"
    local raw_value="${kv#*=}"

    if [[ -z "${!var_name+x}" || -z "${!var_name}" ]]; then
      local expanded
      expanded=$(eval "echo \"${raw_value}\"")
      export "${var_name}=${expanded}"
    fi
  done < "${example_file}"
}

load_example_defaults

IMAGE_NAME="${IMAGE_NAME:-${PROJECT_NAME}:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-${PROJECT_NAME}}"
IP="${1:-${HOST:-0.0.0.0}}"
: "${HTTP_PORT:=8080}"
PORT="${2:-${PUBLISH_PORT:-${HTTP_PORT}}}"

: "${MODEL_PATH:?MODEL_PATH must be set (via .env or .env.example).}"
: "${SERVED_MODEL_NAME:?SERVED_MODEL_NAME must be set (via .env or .env.example).}"
: "${TP_SIZE:?TP_SIZE must be set (via .env or .env.example).}"
: "${MEM_FRACTION_STATIC:?MEM_FRACTION_STATIC must be set (via .env or .env.example).}"
: "${CONTEXT_LENGTH:?CONTEXT_LENGTH must be set (via .env or .env.example).}"
: "${MAX_RUNNING_REQUESTS:?MAX_RUNNING_REQUESTS must be set (via .env or .env.example).}"
: "${MAX_QUEUED_REQUESTS:?MAX_QUEUED_REQUESTS must be set (via .env or .env.example).}"
: "${SHM_SIZE:?SHM_SIZE must be set (via .env or .env.example).}"
: "${GPU_DEVICE:?GPU_DEVICE must be set (via .env or .env.example).}"
: "${RESTART_POLICY:?RESTART_POLICY must be set (via .env or .env.example).}"
: "${HF_HOME:=${HOME}/.cache/huggingface}"
: "${HF_HUB_CACHE:=${HF_HOME}/hub}"
: "${TRANSFORMERS_CACHE:=${HF_HUB_CACHE}}"
if [[ -z "${API_KEY:-}" || "${API_KEY}" == "change-me" || "${API_KEY}" == "change-me-very-strong-token" ]]; then
  echo "ERROR: API_KEY must be set to a private, non-placeholder value in .env" >&2
  return 1 2>/dev/null || exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN is not set; model downloads from the Hugging Face Hub may be rate-limited." >&2
fi

if [[ -n "${TORCHINDUCTOR_COMPILE_THREADS:-}" && ! "${TORCHINDUCTOR_COMPILE_THREADS}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: TORCHINDUCTOR_COMPILE_THREADS must be an integer when set." >&2
  return 1 2>/dev/null || exit 1
fi

mkdir -p "${HF_HUB_CACHE}"

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

DOCKER_TTY_ARGS=()
if [[ -t 0 && -t 1 ]]; then
  DOCKER_TTY_ARGS=(-it)
fi

DOCKER_RUN_CLEANUP_ARGS=(--rm)
if [[ "${RESTART_POLICY}" != "no" ]]; then
  DOCKER_RUN_CLEANUP_ARGS=(--restart "${RESTART_POLICY}")
fi

DOCKER_OPTIONAL_ENV_ARGS=()
if [[ -n "${TORCHINDUCTOR_COMPILE_THREADS:-}" ]]; then
  DOCKER_OPTIONAL_ENV_ARGS+=(
    -e TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS}"
  )
fi

if [[ -n "${CHUNKED_PREFILL_SIZE:-}" ]]; then
  DOCKER_OPTIONAL_ENV_ARGS+=(
    -e CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE}"
  )
fi

if [[ -n "${MAX_PREFILL_TOKENS:-}" ]]; then
  DOCKER_OPTIONAL_ENV_ARGS+=(
    -e MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS}"
  )
fi

echo "Starting container '${CONTAINER_NAME}' with restart policy '${RESTART_POLICY}'"

exec docker run "${DOCKER_RUN_CLEANUP_ARGS[@]}" "${DOCKER_TTY_ARGS[@]}" \
  --name "${CONTAINER_NAME}" \
  --gpus "${GPU_DEVICE}" \
  --shm-size "${SHM_SIZE}" \
  -p "${IP}:${PORT}:${HTTP_PORT}" \
  -e MODEL_PATH="${MODEL_PATH}" \
  -e SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" \
  -e TP_SIZE="${TP_SIZE}" \
  -e MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC}" \
  -e CONTEXT_LENGTH="${CONTEXT_LENGTH}" \
  -e MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS}" \
  -e MAX_QUEUED_REQUESTS="${MAX_QUEUED_REQUESTS}" \
  -e KV_CACHE_DTYPE="${KV_CACHE_DTYPE}" \
  -e ENABLE_TOOLS="${ENABLE_TOOLS}" \
  -e ENABLE_MTP="${ENABLE_MTP}" \
  -e ENABLE_SLEEP_ON_IDLE="${ENABLE_SLEEP_ON_IDLE:-0}" \
  -e TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}" \
  -e ATTENTION_BACKEND="${ATTENTION_BACKEND:-}" \
  -e DISABLE_PREFILL_CUDA_GRAPH="${DISABLE_PREFILL_CUDA_GRAPH:-0}" \
  -e MAMBA_FULL_MEMORY_RATIO="${MAMBA_FULL_MEMORY_RATIO:-}" \
  -e TOOL_SERVER="${TOOL_SERVER:-}" \
  -e SPECULATIVE_ALGORITHM="${SPECULATIVE_ALGORITHM:-NEXTN}" \
  -e SPECULATIVE_DRAFT_MODEL_PATH="${SPECULATIVE_DRAFT_MODEL_PATH:-}" \
  -e SPECULATIVE_NUM_STEPS="${SPECULATIVE_NUM_STEPS:-3}" \
  -e SPECULATIVE_EAGLE_TOPK="${SPECULATIVE_EAGLE_TOPK:-1}" \
  -e SPECULATIVE_NUM_DRAFT_TOKENS="${SPECULATIVE_NUM_DRAFT_TOKENS:-4}" \
  "${DOCKER_OPTIONAL_ENV_ARGS[@]}" \
  -e SGLANG_ENABLE_SPEC_V2="${SGLANG_ENABLE_SPEC_V2}" \
  -e MAMBA_RADIX_CACHE_STRATEGY="${MAMBA_RADIX_CACHE_STRATEGY}" \
  -e MAMBA_SSM_DTYPE="${MAMBA_SSM_DTYPE:-}" \
  -e MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND:-flashinfer_cutlass}" \
  -e API_KEY="${API_KEY}" \
  -e ADMIN_API_KEY="${ADMIN_API_KEY:-${API_KEY}}" \
  -e HTTP_PORT="${HTTP_PORT}" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e HF_HOME="/app/models" \
  -e HF_HUB_CACHE="/app/models/hub" \
  -e TRANSFORMERS_CACHE="/app/models/hub" \
  -v "${HF_HOME}:/app/models" \
  "${IMAGE_NAME}"
