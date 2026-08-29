ARG BASE_IMAGE=lmsysorg/sglang:latest-runtime
FROM ${BASE_IMAGE}

ARG UPGRADE_TRANSFORMERS=0
ARG UPGRADE_SGLANG=0
ARG REQUIRE_QWEN4_EXP=0
ARG SGLANG_GIT_REF=main
ARG SGLANG_WHL_INDEX=https://docs.sglang.ai/whl/cu130/

COPY patch_sglang_transformers_registry.py /usr/local/bin/patch-sglang-transformers-registry

RUN if [ "${UPGRADE_SGLANG}" = "1" ]; then \
      git clone https://github.com/sgl-project/sglang.git /tmp/sglang-source; \
      cd /tmp/sglang-source; \
      git fetch origin "${SGLANG_GIT_REF}"; \
      git checkout --detach FETCH_HEAD; \
      pip install --no-cache-dir --upgrade "/tmp/sglang-source/python[all]" \
        --extra-index-url "${SGLANG_WHL_INDEX}"; \
      rm -rf /tmp/sglang-source; \
    else \
      echo "Skipping sglang source install; using base-image prebuilt binaries."; \
    fi

# Install Transformers last. SGLang declares a Transformers dependency, so
# installing SGLang afterwards can otherwise downgrade a source checkout and
# silently remove newly registered model types such as qwen4_exp.
RUN if [ "${UPGRADE_TRANSFORMERS}" = "1" ]; then \
      pip install --no-cache-dir --upgrade "git+https://github.com/huggingface/transformers.git"; \
      python -c 'from transformers import AutoConfig; AutoConfig.for_model("qwen4_exp")'; \
      python /usr/local/bin/patch-sglang-transformers-registry; \
      python -c 'import sglang.srt.configs'; \
      if [ "${REQUIRE_QWEN4_EXP}" = "1" ]; then python -c 'from sglang.srt.models.registry import ModelRegistry; assert "Qwen4ExpForConditionalGeneration" in ModelRegistry.models, "SGLang image does not implement Qwen4ExpForConditionalGeneration"'; fi; \
    else \
      echo "Skipping transformers upgrade to keep base-image CUDA stack coherent."; \
    fi

ENV MODEL_PATH="nvidia/Qwen3.6-35B-A3B-NVFP4" \
    SERVED_MODEL_NAME="qwen3.6" \
    TP_SIZE="1" \
    MEM_FRACTION_STATIC="0.5" \
    CONTEXT_LENGTH="262144" \
    MAX_RUNNING_REQUESTS="8" \
    MAX_QUEUED_REQUESTS="12" \
    USE_SGLANG_DEFAULTS="0" \
    HTTP_PORT="8080" \
    KV_CACHE_DTYPE="fp8_e4m3" \
    ENABLE_TOOLS="1" \
    TOOL_CALL_PARSER="qwen3_coder" \
    ENABLE_MTP="1" \
    TRUST_REMOTE_CODE="0" \
    REASONING_PARSER="qwen3" \
    ATTENTION_BACKEND="" \
    LINEAR_ATTN_PREFILL_BACKEND="" \
    LINEAR_ATTN_DECODE_BACKEND="" \
    DISABLE_PREFILL_CUDA_GRAPH="0" \
    MAMBA_FULL_MEMORY_RATIO="" \
    ENABLE_SLEEP_ON_IDLE="1" \
    TOOL_SERVER="" \
    SPECULATIVE_ALGORITHM="NEXTN" \
    SPECULATIVE_DRAFT_MODEL_PATH="" \
    SPECULATIVE_NUM_STEPS="3" \
    SPECULATIVE_EAGLE_TOPK="1" \
    SPECULATIVE_NUM_DRAFT_TOKENS="4" \
    SGLANG_ENABLE_SPEC_V2="1" \
    MAMBA_RADIX_CACHE_STRATEGY="extra_buffer" \
    MAMBA_SSM_DTYPE="" \
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
