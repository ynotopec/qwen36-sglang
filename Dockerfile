ARG BASE_IMAGE=lmsysorg/sglang:dev-cu13
FROM ${BASE_IMAGE}

ARG UPGRADE_TRANSFORMERS=0
ARG UPGRADE_SGLANG=0
ARG SGLANG_WHL_INDEX=https://docs.sglang.ai/whl/cu130/

RUN if [ "${UPGRADE_TRANSFORMERS}" = "1" ]; then \
      pip install --no-cache-dir --upgrade "git+https://github.com/huggingface/transformers.git"; \
    else \
      echo "Skipping transformers upgrade to keep base-image CUDA stack coherent."; \
    fi

RUN if [ "${UPGRADE_SGLANG}" = "1" ]; then \
      pip install --no-cache-dir --upgrade \
        sglang sglang[all] sglang-kernel \
        --index-url "${SGLANG_WHL_INDEX}"; \
    else \
      echo "Skipping sglang upgrade; using base-image prebuilt binaries."; \
    fi

ENV MODEL_PATH="Qwen/Qwen3.6-35B-A3B-FP8" \
    SERVED_MODEL_NAME="qwen3.6" \
    TP_SIZE="1" \
    MEM_FRACTION_STATIC="0.5" \
    CONTEXT_LENGTH="262144" \
    MAX_RUNNING_REQUESTS="8" \
    MAX_QUEUED_REQUESTS="8" \
    HTTP_PORT="8080" \
    ENABLE_MULTIMODAL="1" \
    ENABLE_TOOLS="1" \
    ENABLE_MTP="1" \
    TRUST_REMOTE_CODE="0" \
    ATTENTION_BACKEND="" \
    DISABLE_PREFILL_CUDA_GRAPH="0" \
    MAMBA_FULL_MEMORY_RATIO="" \
    ENABLE_SLEEP_ON_IDLE="0" \
    DISABLE_MTP_WITH_MULTIMODAL="0" \
    DISABLE_CHUNKED_PREFILL_ON_DGX_SPARK="1" \
    TOOL_SERVER="" \
    SPECULATIVE_NUM_STEPS="3" \
    SPECULATIVE_EAGLE_TOPK="1" \
    SPECULATIVE_NUM_DRAFT_TOKENS="4" \
    SGLANG_ENABLE_SPEC_V2="1" \
    MAMBA_RADIX_CACHE_STRATEGY="extra_buffer" \
    MOE_RUNNER_BACKEND="flashinfer_cutlass" \
    HEALTHCHECK_MONITOR_START_GRACE="900" \
    HEALTHCHECK_MONITOR_INTERVAL="15" \
    HEALTHCHECK_MONITOR_MAX_FAILURES="8" \
    HF_HOME="/app/models" \
    TRANSFORMERS_CACHE="/app/models/hub" \
    HF_HUB_CACHE="/app/models/hub" \
    XDG_CACHE_HOME="/app/models" \
    FLASHINFER_DISABLE_VERSION_CHECK="1" \
    PYTHONUNBUFFERED="1"

RUN mkdir -p /app/models /app/data
WORKDIR /app

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=30s --start-period=900s --retries=5 \
  CMD curl -fsS "http://127.0.0.1:${HTTP_PORT}/health" || exit 1

CMD ["/app/entrypoint.sh"]
