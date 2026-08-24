#!/usr/bin/env bash
set -euo pipefail

effective_chunked_prefill_size="${CHUNKED_PREFILL_SIZE:-}"

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
  --max-queued-requests "${MAX_QUEUED_REQUESTS}"
  --reasoning-parser qwen3
  --sampling-defaults model
)

if [[ "${TRUST_REMOTE_CODE:-0}" == "1" ]]; then
  ARGS+=( --trust-remote-code )
fi

if [[ -n "${ATTENTION_BACKEND:-}" ]]; then
  ARGS+=( --attention-backend "${ATTENTION_BACKEND}" )
fi

if [[ "${DISABLE_PREFILL_CUDA_GRAPH:-0}" == "1" ]]; then
  ARGS+=( --disable-prefill-cuda-graph )
fi

if [[ -n "${MAMBA_FULL_MEMORY_RATIO:-}" ]]; then
  ARGS+=( --mamba-full-memory-ratio "${MAMBA_FULL_MEMORY_RATIO}" )
fi

if [[ -n "${effective_chunked_prefill_size}" ]]; then
  ARGS+=( --chunked-prefill-size "${effective_chunked_prefill_size}" )
fi

if [[ -n "${MAX_PREFILL_TOKENS:-}" ]]; then
  ARGS+=( --max-prefill-tokens "${MAX_PREFILL_TOKENS}" )
fi

if [[ "${ENABLE_SLEEP_ON_IDLE:-0}" == "1" ]]; then
  ARGS+=( --sleep-on-idle )
fi

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

if [[ "${ENABLE_TOOLS:-1}" == "1" ]]; then
  ARGS+=( --tool-call-parser qwen3_coder )
  if [[ -n "${TOOL_SERVER:-}" ]]; then
    ARGS+=( --tool-server "${TOOL_SERVER}" )
  fi
fi

if [[ "${ENABLE_MTP:-1}" == "1" ]]; then
  ARGS+=(
    --speculative-algo "${SPECULATIVE_ALGORITHM:-NEXTN}"
    --speculative-num-steps "${SPECULATIVE_NUM_STEPS:-3}"
    --speculative-eagle-topk "${SPECULATIVE_EAGLE_TOPK:-1}"
    --speculative-num-draft-tokens "${SPECULATIVE_NUM_DRAFT_TOKENS:-4}"
    --mamba-radix-cache-strategy "${MAMBA_RADIX_CACHE_STRATEGY:-extra_buffer}"
  )
  if [[ -n "${SPECULATIVE_DRAFT_MODEL_PATH:-}" ]]; then
    ARGS+=( --speculative-draft-model-path "${SPECULATIVE_DRAFT_MODEL_PATH}" )
  fi
fi

if [[ -n "${MAMBA_SSM_DTYPE:-}" ]]; then
  ARGS+=( --mamba-ssm-dtype "${MAMBA_SSM_DTYPE}" )
fi

if [[ "${ENABLE_MIXED_CHUNK:-0}" == "1" ]]; then
  ARGS+=( --enable-mixed-chunk )
fi

if [[ -n "${MOE_RUNNER_BACKEND:-}" ]]; then
  ARGS+=( --moe-runner-backend "${MOE_RUNNER_BACKEND}" )
fi

echo "Launching SGLang:"
printf ' %q' "${ARGS[@]}"
echo

health_url="http://127.0.0.1:${HTTP_PORT}/health"
monitor_interval="${HEALTHCHECK_MONITOR_INTERVAL:-15}"
max_failures="${HEALTHCHECK_MONITOR_MAX_FAILURES:-8}"
start_grace="${HEALTHCHECK_MONITOR_START_GRACE:-900}"

validate_nonnegative_integer() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: ${name} must be a non-negative integer; got '${value}'." >&2
    exit 2
  fi
}

validate_positive_integer() {
  local name="$1"
  local value="$2"
  validate_nonnegative_integer "${name}" "${value}"
  if (( 10#${value} == 0 )); then
    echo "ERROR: ${name} must be greater than zero." >&2
    exit 2
  fi
}

validate_nonnegative_integer HEALTHCHECK_MONITOR_START_GRACE "${start_grace}"
validate_positive_integer HEALTHCHECK_MONITOR_INTERVAL "${monitor_interval}"
validate_positive_integer HEALTHCHECK_MONITOR_MAX_FAILURES "${max_failures}"
start_grace=$((10#${start_grace}))
monitor_interval=$((10#${monitor_interval}))
max_failures=$((10#${max_failures}))

"${ARGS[@]}" &
server_pid=$!

cleanup() {
  trap - INT TERM
  if kill -0 "${server_pid}" 2>/dev/null; then
    kill -TERM "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" || true
  fi
}
trap cleanup INT TERM

(
  start_ts=$(date +%s)
  fails=0
  while kill -0 "${server_pid}" 2>/dev/null; do
    now_ts=$(date +%s)
    if (( now_ts - start_ts < start_grace )); then
      sleep "${monitor_interval}"
      continue
    fi

    if curl -fsS "${health_url}" >/dev/null 2>&1; then
      fails=0
    else
      fails=$((fails + 1))
      echo "[watchdog] /health failed (${fails}/${max_failures}) on ${health_url}" >&2
      if (( fails >= max_failures )); then
        echo "[watchdog] Health check failed repeatedly; terminating sglang pid ${server_pid} to trigger container restart." >&2
        kill -TERM "${server_pid}" 2>/dev/null || true
        sleep 10
        kill -KILL "${server_pid}" 2>/dev/null || true
        break
      fi
    fi
    sleep "${monitor_interval}"
  done
) &
monitor_pid=$!

set +e
wait "${server_pid}"
exit_code=$?
set -e
kill "${monitor_pid}" 2>/dev/null || true
wait "${monitor_pid}" 2>/dev/null || true
exit "${exit_code}"
