ARG BASE_IMAGE=lmsysorg/sglang:dev-cu13
FROM ${BASE_IMAGE}

ARG UPGRADE_TRANSFORMERS=0
ARG UPGRADE_SGLANG=0
ARG SGLANG_WHL_INDEX=https://docs.sglang.ai/whl/cu130/
ARG SGLANG_GIT_REF=refs/pull/20547/head
ARG HUGGINGFACE_HUB_SPEC

RUN if [ "${UPGRADE_TRANSFORMERS}" = "1" ]; then \
      pip install --no-cache-dir --upgrade "git+https://github.com/huggingface/transformers.git"; \
    else \
      echo "Skipping transformers upgrade to keep base-image CUDA stack coherent."; \
    fi

RUN if [ "${UPGRADE_SGLANG}" = "1" ]; then \
      pip install --no-cache-dir --upgrade \
        "git+https://github.com/sgl-project/sglang.git@${SGLANG_GIT_REF}#subdirectory=python"; \
    elif [ "${UPGRADE_SGLANG}" = "wheel" ]; then \
      pip install --no-cache-dir --upgrade \
        sglang sglang[all] sglang-kernel \
        --index-url "${SGLANG_WHL_INDEX}"; \
    else \
      echo "Skipping sglang upgrade; using base-image prebuilt binaries."; \
    fi

RUN if [ "${UPGRADE_SGLANG}" != "0" ] || [ "${UPGRADE_TRANSFORMERS}" = "1" ]; then \
      pip install --no-cache-dir --upgrade "huggingface_hub${HUGGINGFACE_HUB_SPEC:->=0.36.0,<1.0}"; \
    else \
      echo "Skipping huggingface_hub compatibility upgrade."; \
    fi

ENV MODEL_PATH="Qwen/Qwen3.6-35B-A3B-FP8" \
    SERVED_MODEL_NAME="qwen3.6" \
    TP_SIZE="1" \
    MEM_FRACTION_STATIC="0.5" \
    CONTEXT_LENGTH="262144" \
    MAX_RUNNING_REQUESTS="8" \
    MAX_QUEUED_REQUESTS="8" \
    CHUNKED_PREFILL_SIZE="4096" \
    HTTP_PORT="8080" \
    ENABLE_MULTIMODAL="1" \
    ENABLE_TOOLS="1" \
    ENABLE_MTP="1" \
    ENABLE_DFLASH="1" \
    DFLASH_DRAFT_MODEL_PATH="z-lab/Qwen3.6-35B-A3B-DFlash" \
    ATTENTION_BACKEND="fa3" \
    TRUST_REMOTE_CODE="1" \
    TOOL_SERVER="" \
    SPECULATIVE_NUM_STEPS="3" \
    SPECULATIVE_EAGLE_TOPK="1" \
    SPECULATIVE_NUM_DRAFT_TOKENS="16" \
    SGLANG_ENABLE_SPEC_V2="1" \
    SGLANG_ENABLE_DFLASH_SPEC_V2="0" \
    SGLANG_ENABLE_OVERLAP_PLAN_STREAM="0" \
    MAMBA_SCHEDULER_STRATEGY="extra_buffer" \
    HEALTHCHECK_MONITOR_START_GRACE="900" \
    HEALTHCHECK_MONITOR_INTERVAL="15" \
    HEALTHCHECK_MONITOR_MAX_FAILURES="8" \
    HF_HOME="/app/models" \
    TRANSFORMERS_CACHE="/app/models" \
    HF_HUB_CACHE="/app/models" \
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
