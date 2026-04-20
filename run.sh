#!/usr/bin/env bash

# Usage:
#   source ./run.sh [IP] [PORT]
# or:
#   ./run.sh [IP] [PORT]

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="$(basename "${PROJECT_DIR}")"

IMAGE_NAME="${IMAGE_NAME:-${PROJECT_NAME}:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-${PROJECT_NAME}}"

IP="${1:-0.0.0.0}"
PORT="${2:-8080}"

if [[ -f "${PROJECT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${PROJECT_DIR}/.env"
  set +a
fi

: "${MODEL_PATH:=Qwen/Qwen3.6-35B-A3B-FP8}"
: "${SERVED_MODEL_NAME:=qwen3.6-35b-a3b-fp8}"
: "${TP_SIZE:=1}"
: "${MEM_FRACTION_STATIC:=0.80}"
: "${CONTEXT_LENGTH:=131072}"
: "${MAX_RUNNING_REQUESTS:=32}"
: "${CHUNKED_PREFILL_SIZE:=4096}"
: "${KV_CACHE_DTYPE:=}"
: "${ENABLE_MULTIMODAL:=1}"
: "${ENABLE_TOOLS:=1}"
: "${ENABLE_MTP:=0}"
: "${HF_HOME:=${HOME}/.cache/huggingface}"
: "${HF_HUB_CACHE:=${HF_HOME}}"
: "${TRANSFORMERS_CACHE:=${HF_HOME}}"
: "${HTTP_PORT:=${PORT}}"
: "${SHM_SIZE:=16g}"
: "${GPU_DEVICE:=all}"
: "${SGLANG_ENABLE_SPEC_V2:=0}"
: "${MAMBA_SCHEDULER_STRATEGY:=no_buffer}"

if [[ -z "${API_KEY:-}" ]]; then
  echo "ERROR: API_KEY is required in .env" >&2
  return 1 2>/dev/null || exit 1
fi

mkdir -p "${HF_HOME}"

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

DOCKER_TTY_ARGS=()
if [[ -t 0 && -t 1 ]]; then
  DOCKER_TTY_ARGS=(-it)
fi

exec docker run --rm "${DOCKER_TTY_ARGS[@]}" \
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
  -e CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE}" \
  -e KV_CACHE_DTYPE="${KV_CACHE_DTYPE}" \
  -e ENABLE_MULTIMODAL="${ENABLE_MULTIMODAL}" \
  -e ENABLE_TOOLS="${ENABLE_TOOLS}" \
  -e ENABLE_MTP="${ENABLE_MTP}" \
  -e TOOL_SERVER="${TOOL_SERVER:-}" \
  -e SPECULATIVE_NUM_STEPS="${SPECULATIVE_NUM_STEPS:-3}" \
  -e SPECULATIVE_EAGLE_TOPK="${SPECULATIVE_EAGLE_TOPK:-1}" \
  -e SPECULATIVE_NUM_DRAFT_TOKENS="${SPECULATIVE_NUM_DRAFT_TOKENS:-4}" \
  -e SGLANG_ENABLE_SPEC_V2="${SGLANG_ENABLE_SPEC_V2}" \
  -e MAMBA_SCHEDULER_STRATEGY="${MAMBA_SCHEDULER_STRATEGY}" \
  -e API_KEY="${API_KEY}" \
  -e ADMIN_API_KEY="${ADMIN_API_KEY:-}" \
  -e HTTP_PORT="${HTTP_PORT}" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e HF_HOME="/app/models" \
  -e HF_HUB_CACHE="/app/models" \
  -e TRANSFORMERS_CACHE="/app/models" \
  -v "${HF_HOME}:/app/models" \
  "${IMAGE_NAME}"
