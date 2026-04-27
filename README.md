# sglang-qwen36

Serve `Qwen/Qwen3.6-35B-A3B-FP8` with SGLang in Docker.

## Features
- idempotent `install.sh`
- `source run.sh [IP] [PORT]`
- API token required
- compatible H100 / DGX Spark
- optional multimodal, tools, and MTP

## Requirements
- Docker
- NVIDIA Container Toolkit
- NVIDIA GPU
- enough VRAM / memory headroom

## Install
```bash
./install.sh
````

If `.env` does not exist, it is created from `.env.example`.

## Configure

Edit `.env` and set at least:

```bash
API_KEY=change-me
```

Optional:

* `HF_TOKEN` if the model download requires authentication
* `ADMIN_API_KEY` for admin endpoints
* `ENABLE_MTP=1` to enable speculative decoding / MTP
* `TOOL_SERVER=...` if using tool execution
* `KV_CACHE_DTYPE=...` to override KV cache precision (for example `auto`, `fp8_e4m3`, or `fp8_e5m2`)
* `FLASHINFER_DISABLE_VERSION_CHECK=0` if you want to re-enable strict `flashinfer`/`flashinfer-jit-cache` version checks
* `UPGRADE_SGLANG=1` only if you explicitly want to replace base-image SGLang binaries (off by default for CUDA compatibility)
* `UPGRADE_TRANSFORMERS=1` only if you explicitly want a bleeding-edge `transformers` build

If you omit `HF_TOKEN`, Hugging Face may repeatedly print unauthenticated/rate-limit warnings during model download.

## Run

```bash
source ./run.sh 0.0.0.0 8080
```

## Health

```bash
curl http://127.0.0.1:8080/health
```

## OpenAI-compatible API

```bash
curl http://127.0.0.1:8080/v1/models \
  -H "Authorization: Bearer ${API_KEY}"
```

### Chat completion

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d '{
    "model": "qwen3.6-35b-a3b-fp8",
    "messages": [
      {"role": "user", "content": "Bonjour"}
    ]
  }'
```

## Suggested defaults

### H100

* `TP_SIZE=1`
* `MEM_FRACTION_STATIC=0.80`
* `CONTEXT_LENGTH=131072`
* `MAX_RUNNING_REQUESTS=32`

### DGX Spark

Start conservatively:

* `TP_SIZE=1`
* `MEM_FRACTION_STATIC=0.72`
* `CONTEXT_LENGTH=65536`
* `MAX_RUNNING_REQUESTS=8`

Then increase gradually.

## Notes

* SGLang supports `--api-key`, `--admin-api-key`, `--reasoning-parser`, `--tool-call-parser`, `--enable-multimodal`, and speculative decoding flags.
* Qwen provides SGLang recommendations for this FP8 model, including `reasoning-parser qwen3`, `tool-call-parser qwen3_coder`, and MTP-related flags.
* If you hit OOM, reduce `MEM_FRACTION_STATIC`, `CONTEXT_LENGTH`, or `MAX_RUNNING_REQUESTS`.
* If you see `python -m sglang.launch_server is still supported`, update your startup command to `sglang serve`.
* Warnings like `Unexpected error during package walk: cutlass.cute.experimental` are generally non-fatal in current SGLang/CUTLASS combinations.
* This image defaults `FLASHINFER_DISABLE_VERSION_CHECK=1` to avoid startup failures caused by transient package skew in upstream base images.
* `run.sh` now defaults to `RESTART_POLICY=unless-stopped`, so the container auto-restarts when SGLang hangs or crashes. Set `RESTART_POLICY=no` to keep the previous one-shot `--rm` behavior.

## Troubleshooting: `/health` returns 503 with detokenizer heartbeat timeout

If logs repeatedly show messages like:

* `Health check failed. Server couldn't get a response from detokenizer...`
* `GET /health ... 503 Service Unavailable`

while `/v1/models` can still return `200`, it usually means an internal SGLang worker got stuck (often after driver/CUDA stack changes) rather than an API-key/network issue.

Recommended actions:

1. Keep the default `RESTART_POLICY=unless-stopped` so Docker restarts the container automatically on process failure.
2. Rebuild and run with conservative memory settings first (for example lower `MEM_FRACTION_STATIC` and `MAX_RUNNING_REQUESTS`), then scale up.
3. If the issue started right after a CUDA or base-image update, pin your previously known-good image/tag instead of tracking moving `dev` tags.
4. If startup fails with `libnvrtc.so.12` / `Could not load any common_ops library`, rebuild without Python package overrides so base-image CUDA-matched binaries are kept (default behavior in this repo now).

### CUDA stack mismatch quick check

If logs show both:

* `Compute capability ... 12.1` not supported by current PyTorch build, and/or
* `ImportError: libnvrtc.so.12` while runtime reports CUDA 13

you likely have a mixed wheel stack (for example, upgraded `sglang-kernel` built for a different CUDA ABI).
This repository now avoids overriding SGLang by default; only opt-in to upgrades when intentionally testing a matching wheel index.

[1]: https://hub.docker.com/r/lmsysorg/sglang/tags "lmsysorg/sglang - Docker Image"
[2]: https://github.com/sgl-project/sglang/issues/20973 "[Bug] can't load AxionML/Qwen3.5-35B-A3B-NVFP4 on fresh `lmsysorg/sglang:dev-cu13` on Nvidia DGX Spark · Issue #20973 · sgl-project/sglang · GitHub"
[3]: https://docs.sglang.ai/advanced_features/server_arguments.html "Server Arguments — SGLang"
