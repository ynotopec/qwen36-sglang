#!/usr/bin/env bash
set -euo pipefail

effective_mem_fraction_static="${MEM_FRACTION_STATIC}"
effective_context_length="${CONTEXT_LENGTH}"
effective_max_running_requests="${MAX_RUNNING_REQUESTS}"
effective_chunked_prefill_size="${CHUNKED_PREFILL_SIZE}"
effective_kv_cache_dtype="${KV_CACHE_DTYPE:-}"
effective_attention_backend="${ATTENTION_BACKEND:-}"
effective_enable_prefix_caching="${ENABLE_PREFIX_CACHING:-}"
effective_max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS:-}"
effective_enable_mtp="${ENABLE_MTP:-1}"

gpu_name="${GPU_NAME_OVERRIDE:-}"
if [[ -z "${gpu_name}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || true)
fi

is_dgx_spark=0
if [[ "${gpu_name}" =~ (DGX[[:space:]]*Spark|GB10) ]]; then
  is_dgx_spark=1
fi

if [[ "${DGX_SPARK_OPTIMIZE:-1}" == "1" && "${is_dgx_spark}" == "1" ]]; then
  if [[ "${effective_mem_fraction_static}" == "0.5" ]]; then
    effective_mem_fraction_static="0.72"
  fi
  if [[ "${effective_context_length}" == "262144" ]]; then
    effective_context_length="65536"
  fi
  if [[ -z "${effective_kv_cache_dtype}" ]]; then
    effective_kv_cache_dtype="fp8_e4m3"
  fi
  if [[ -z "${effective_attention_backend}" ]]; then
    effective_attention_backend="flashinfer"
  fi
  if [[ -z "${effective_enable_prefix_caching}" ]]; then
    effective_enable_prefix_caching="1"
  fi
  if [[ -z "${effective_max_num_batched_tokens}" ]]; then
    effective_max_num_batched_tokens="16384"
  fi
  if [[ "${DGX_SPARK_DISABLE_MTP:-1}" == "1" && "${effective_enable_mtp}" == "1" ]]; then
    effective_enable_mtp="0"
  fi
  echo "Detected DGX Spark/GB10; applying DGX Spark profile: mem_fraction_static=${effective_mem_fraction_static}, context_length=${effective_context_length}, max_running_requests=${effective_max_running_requests}, kv_cache_dtype=${effective_kv_cache_dtype}, attention_backend=${effective_attention_backend}, enable_prefix_caching=${effective_enable_prefix_caching}, max_num_batched_tokens=${effective_max_num_batched_tokens}, enable_mtp=${effective_enable_mtp}. Set DGX_SPARK_OPTIMIZE=0 or override individual variables to disable parts of this profile." >&2
fi

if [[ "${ENABLE_MULTIMODAL:-1}" == "1" && "${DISABLE_CHUNKED_PREFILL_ON_DGX_SPARK:-1}" == "1" ]]; then
  if [[ "${is_dgx_spark}" == "1" ]]; then
    if [[ "${effective_chunked_prefill_size}" != "-1" ]]; then
      echo "Detected DGX Spark/GB10 with multimodal enabled; disabling chunked prefill to avoid SGLang mamba-cache crashes on image prompts. Set DISABLE_CHUNKED_PREFILL_ON_DGX_SPARK=0 to force CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE}." >&2
    fi
    effective_chunked_prefill_size="-1"
  fi
fi

ARGS=(
  sglang serve
  --host 0.0.0.0
  --port "${HTTP_PORT}"
  --model-path "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --tp-size "${TP_SIZE}"
  --mem-fraction-static "${effective_mem_fraction_static}"
  --context-length "${effective_context_length}"
  --max-running-requests "${effective_max_running_requests}"
  --max-queued-requests "${MAX_QUEUED_REQUESTS}"
  --chunked-prefill-size "${effective_chunked_prefill_size}"
  --reasoning-parser qwen3
  --sampling-defaults model
  --sleep-on-idle
)

if [[ -n "${effective_attention_backend}" ]]; then
  ARGS+=( --attention-backend "${effective_attention_backend}" )
fi

if [[ "${effective_enable_prefix_caching}" == "1" ]]; then
  ARGS+=( --enable-prefix-caching )
fi

if [[ -n "${effective_max_num_batched_tokens}" ]]; then
  ARGS+=( --max-num-batched-tokens "${effective_max_num_batched_tokens}" )
fi

if [[ -n "${effective_kv_cache_dtype}" ]]; then
  case "${effective_kv_cache_dtype}" in
    fp8)
      echo "KV_CACHE_DTYPE=fp8 is not a valid SGLang CLI value; using fp8_e4m3." >&2
      ARGS+=( --kv-cache-dtype fp8_e4m3 )
      ;;
    auto|fp8_e5m2|fp8_e4m3|bf16|bfloat16|fp4_e2m1)
      ARGS+=( --kv-cache-dtype "${effective_kv_cache_dtype}" )
      ;;
    *)
      echo "ERROR: Unsupported KV_CACHE_DTYPE='${effective_kv_cache_dtype}'." >&2
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

if [[ "${effective_enable_mtp}" == "1" && "${ENABLE_MULTIMODAL:-1}" == "1" && "${DISABLE_MTP_WITH_MULTIMODAL:-0}" == "1" ]]; then
  echo "ENABLE_MTP=1 with ENABLE_MULTIMODAL=1 can crash SGLang mamba cache during chunked image prefill; disabling MTP. Set DISABLE_MTP_WITH_MULTIMODAL=0 to force it." >&2
elif [[ "${effective_enable_mtp}" == "1" ]]; then
  ARGS+=(
    --speculative-algo NEXTN
    --speculative-num-steps "${SPECULATIVE_NUM_STEPS:-3}"
    --speculative-eagle-topk "${SPECULATIVE_EAGLE_TOPK:-1}"
    --speculative-num-draft-tokens "${SPECULATIVE_NUM_DRAFT_TOKENS:-4}"
    --mamba-scheduler-strategy "${MAMBA_SCHEDULER_STRATEGY:-extra_buffer}"
  )
fi

if [[ "${ENABLE_MIXED_CHUNK:-0}" == "1" ]]; then
  ARGS+=( --enable-mixed-chunk )
fi

echo "Launching SGLang:"
printf ' %q' "${ARGS[@]}"
echo

health_url="http://127.0.0.1:${HTTP_PORT}/health"
monitor_interval="${HEALTHCHECK_MONITOR_INTERVAL:-15}"
max_failures="${HEALTHCHECK_MONITOR_MAX_FAILURES:-8}"
start_grace="${HEALTHCHECK_MONITOR_START_GRACE:-900}"

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

wait "${server_pid}"
exit_code=$?
kill "${monitor_pid}" 2>/dev/null || true
wait "${monitor_pid}" 2>/dev/null || true
exit "${exit_code}"
