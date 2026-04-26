ARG BASE_IMAGE=lmsysorg/sglang:dev-cu13
FROM ${BASE_IMAGE}

RUN pip install --no-cache-dir --upgrade \
    "git+https://github.com/huggingface/transformers.git"
#RUN pip install --no-cache-dir --upgrade \
#    "huggingface_hub[cli]>=0.30.0"
RUN pip install --no-cache-dir --upgrade \
    sglang sglang[all]

ENV MODEL_PATH="Qwen/Qwen3.6-35B-A3B-FP8" \
    SERVED_MODEL_NAME="qwen3.6-35b-a3b-fp8" \
    TP_SIZE="1" \
    MEM_FRACTION_STATIC="0.80" \
    CONTEXT_LENGTH="131072" \
    MAX_RUNNING_REQUESTS="32" \
    CHUNKED_PREFILL_SIZE="4096" \
    HTTP_PORT="8080" \
    ENABLE_MULTIMODAL="1" \
    ENABLE_TOOLS="1" \
    ENABLE_MTP="0" \
    TOOL_SERVER="" \
    SPECULATIVE_NUM_STEPS="3" \
    SPECULATIVE_EAGLE_TOPK="1" \
    SPECULATIVE_NUM_DRAFT_TOKENS="4" \
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
