#!/usr/bin/env bash
set -euo pipefail

ARGS=(
  sglang serve
  --host 0.0.0.0
  --port "${HTTP_PORT}"
  --model-path "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --tp-size "${TP_SIZE}"
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
  --context-length "${CONTEXT_LENGTH}"
  --max-running-requests "${MAX_RUNNING_REQUESTS}"
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}"
  --reasoning-parser qwen3
  --sampling-defaults model
  --sleep-on-idle
)

if [[ -n "${KV_CACHE_DTYPE:-}" ]]; then
  case "${KV_CACHE_DTYPE}" in
    fp8)
      echo "KV_CACHE_DTYPE=fp8 is not a valid SGLang CLI value; using fp8_e4m3." >&2
      ARGS+=( --kv-cache-dtype fp8_e4m3 )
      ;;
    auto|fp8_e5m2|fp8_e4m3|bf16|bfloat16|fp4_e2m1)
      ARGS+=( --kv-cache-dtype "${KV_CACHE_DTYPE}" )
      ;;
    *)
      echo "ERROR: Unsupported KV_CACHE_DTYPE='${KV_CACHE_DTYPE}'." >&2
      echo "Supported values: auto, fp8_e5m2, fp8_e4m3, bf16, bfloat16, fp4_e2m1." >&2
      exit 2
      ;;
  esac
fi

if [[ -n "${API_KEY:-}" ]]; then
  ARGS+=( --api-key "${API_KEY}" )
fi

if [[ -n "${ADMIN_API_KEY:-}" ]]; then
  ARGS+=( --admin-api-key "${ADMIN_API_KEY}" )
fi

if [[ "${ENABLE_MULTIMODAL:-1}" == "1" ]]; then
  ARGS+=( --enable-multimodal )
fi

if [[ "${ENABLE_TOOLS:-1}" == "1" ]]; then
  ARGS+=( --tool-call-parser qwen3_coder )
  if [[ -n "${TOOL_SERVER:-}" ]]; then
    ARGS+=( --tool-server "${TOOL_SERVER}" )
  fi
fi

if [[ "${ENABLE_MTP:-0}" == "1" ]]; then
  ARGS+=(
    --speculative-algo NEXTN
    --speculative-num-steps "${SPECULATIVE_NUM_STEPS:-3}"
    --speculative-eagle-topk "${SPECULATIVE_EAGLE_TOPK:-1}"
    --speculative-num-draft-tokens "${SPECULATIVE_NUM_DRAFT_TOKENS:-4}"
    --mamba-scheduler-strategy "${MAMBA_SCHEDULER_STRATEGY:-extra_buffer}"
  )
fi

echo "Launching SGLang:"
printf ' %q' "${ARGS[@]}"
echo

exec "${ARGS[@]}"
