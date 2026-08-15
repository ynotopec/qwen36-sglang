# sglang-qwen36

Serve `Qwen/Qwen3.6-35B-A3B-FP8` with SGLang in Docker. An optional,
dependency-free `RadixArk/Qwen3.8-27B-NVFP4` profile is documented in
`.env.example` without changing the existing defaults.

## Features
- idempotent `install.sh`
- `source run.sh [IP] [PORT]`
- API token required
- compatible H100 / DGX Spark
- optional multimodal, tools, and MTP
- optional Qwen3.8 NVFP4 configuration using the packages from the base image

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

`install.sh` and `run.sh` now treat `.env.example` as the default source of truth: any unset variable is loaded from commented defaults in `.env.example`, and `.env` only overrides what you change. This includes `BASE_IMAGE`, so you can pin the upstream SGLang Docker image/tag in `.env` after copying it from `.env.example`.

## Configure

Edit `.env` and set at least:

```bash
API_KEY=change-me
```

Optional:

* `HF_TOKEN` if the model download requires authentication
* `ADMIN_API_KEY` for admin endpoints
* `ENABLE_MTP=1` to enable speculative decoding / MTP
* `MAMBA_RADIX_CACHE_STRATEGY=extra_buffer` to tune MTP mamba radix-cache scheduling without using the deprecated SGLang scheduler flag
* `MOE_RUNNER_BACKEND=flashinfer_cutlass` selects the NVFP4-compatible FlashInfer MoE backend (the wrapper default); override it only when your model and SGLang build support another backend
* `DISABLE_MTP_WITH_MULTIMODAL=0` keeps MTP enabled by default, including text-only requests on a multimodal server; set `1` for an image-safe server profile that disables process-wide MTP
* `CHUNKED_PREFILL_SIZE=4096` to pass `--chunked-prefill-size`; leave it unset to omit the SGLang flag
* `ATTENTION_BACKEND=flashinfer`, `DISABLE_PREFILL_CUDA_GRAPH=1`, and `MAMBA_FULL_MEMORY_RATIO=4.59` expose the remaining Qwen3.8 launch settings
* `DISABLE_CHUNKED_PREFILL_ON_DGX_SPARK=1` automatically changes `--chunked-prefill-size` to `-1` on DGX Spark/GB10 multimodal servers when `CHUNKED_PREFILL_SIZE` is set, while leaving H100 defaults unchanged
* `ENABLE_MIXED_CHUNK=1` to enable SGLang mixed-chunk scheduling (`0` by default)
* `ENABLE_SLEEP_ON_IDLE=1` to opt in to SGLang `--sleep-on-idle` (`0` by default)
* `TOOL_SERVER=...` if using tool execution
* `KV_CACHE_DTYPE=...` to override KV cache precision (for example `auto`, `fp8_e4m3`, or `fp8_e5m2`)
* `FLASHINFER_DISABLE_VERSION_CHECK=0` if you want to re-enable strict `flashinfer`/`flashinfer-jit-cache` version checks
* `BASE_IMAGE=lmsysorg/sglang:dev-cu13` to select or pin the upstream SGLang Docker image/tag used by `install.sh`
* `UPGRADE_SGLANG=1` only if you explicitly want to replace base-image SGLang binaries (off by default for CUDA compatibility)
* `UPGRADE_TRANSFORMERS=1` only if you explicitly want a bleeding-edge `transformers` build

If you omit `HF_TOKEN`, Hugging Face may repeatedly print unauthenticated/rate-limit warnings during model download.

Hub downloads are stored under `${HOME}/.cache/huggingface/hub` and mounted as
`/app/models/hub` in the container.

The optional Qwen3.8 profile is grouped at the end of `.env.example`. Copy that
block into `.env` and remove the space after each `#`; these example lines are
deliberately excluded from the default loader.

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
    "model": "qwen3.6",
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
* `Using default W8A8 Block FP8 kernel config ... Config file not found ...` is also non-fatal: SGLang falls back to a safe default FP8 kernel. You can ignore it for first boot, or run SGLang's FP8 tuning workflow to generate a device-specific config for better throughput.
* This image defaults `FLASHINFER_DISABLE_VERSION_CHECK=1` to avoid startup failures caused by transient package skew in upstream base images.
* `run.sh` now defaults to `RESTART_POLICY=unless-stopped`, so the container auto-restarts when SGLang hangs or crashes. Set `RESTART_POLICY=no` to keep the previous one-shot `--rm` behavior.

## Troubleshooting: unsupported NVFP4 MoE backend

If startup fails during FlashInfer autotuning with:

```text
NotImplementedError: Unsupported moe_runner_backend for NVFP4 MoE: MoeRunnerBackend.FLASHINFER_TRTLLM
```

SGLang selected its TensorRT-LLM MoE runner, which does not implement the NVFP4 path. This wrapper explicitly passes `--moe-runner-backend flashinfer_cutlass` by default. Rebuild the image and recreate the container so the updated entrypoint is used. You can configure the value with `MOE_RUNNER_BACKEND`, but NVFP4 models should keep `flashinfer_cutlass`.

## Troubleshooting: CUDA `indexSelectSmallIndex` assert on image input

On DGX Spark, an image request can create a long multimodal prefill that is chunked at `CHUNKED_PREFILL_SIZE`. If speculative MTP is also enabled, current SGLang builds can fail inside the mamba cache while stashing that unfinished chunk. Logs typically include:

```text
indexSelectSmallIndex: Assertion `srcIndex < srcSelectDimSize` failed
mamba_radix_cache.py ... donate_mamba_ping_pong_slot
CUDA error: device-side assert triggered
```

This wrapper now applies a DGX Spark-specific default workaround before SGLang starts: when `ENABLE_MULTIMODAL=1`, `DISABLE_CHUNKED_PREFILL_ON_DGX_SPARK=1`, and the detected GPU name contains `DGX Spark` or `GB10`, it launches SGLang with `--chunked-prefill-size -1`. SGLang documents `-1` as the way to disable chunked prefill, and this targets the failing path shown in the stack trace: `stash_chunked_request` -> `mamba_radix_cache.py` -> `donate_mamba_ping_pong_slot`.

MTP is disabled in the Qwen3.8 profile. If you opt in with `ENABLE_MTP=1`, it remains enabled with multimodal requests unless you also set `DISABLE_MTP_WITH_MULTIMODAL=1`. If DGX Spark image requests fail after chunked prefill is disabled, use that stronger fallback to disable process-wide MTP as well.

H100 and DGX Spark do not exercise identical runtime paths: H100 uses a mature Hopper CUDA/kernel stack with dedicated HBM, while DGX Spark uses a newer Blackwell-class CUDA path and tighter memory behavior. The automatic chunked-prefill workaround is therefore scoped to detected DGX Spark/GB10 systems and leaves H100 defaults unchanged. To force the original behavior on DGX Spark, set `DISABLE_CHUNKED_PREFILL_ON_DGX_SPARK=0`.

## Troubleshooting: `TORCHINDUCTOR_COMPILE_THREADS` parse errors

If startup fails with:

```text
ValueError: invalid literal for int() with base 10: ''
```

make sure `TORCHINDUCTOR_COMPILE_THREADS` is either unset/commented out or set to an integer such as `2`. `run.sh` intentionally omits the variable from `docker run` when it is empty because PyTorch treats an empty-but-present environment variable as invalid.

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
5. This image now includes an internal watchdog in `entrypoint.sh`: if `/health` keeps failing after startup grace, it terminates SGLang so Docker restart policy can recover automatically. Tune with `HEALTHCHECK_MONITOR_START_GRACE`, `HEALTHCHECK_MONITOR_INTERVAL`, and `HEALTHCHECK_MONITOR_MAX_FAILURES`.

### CUDA stack mismatch quick check

If logs show both:

* `Compute capability ... 12.1` not supported by current PyTorch build, and/or
* `ImportError: libnvrtc.so.12` while runtime reports CUDA 13

you likely have a mixed wheel stack (for example, upgraded `sglang-kernel` built for a different CUDA ABI).
This repository now avoids overriding SGLang by default; only opt-in to upgrades when intentionally testing a matching wheel index.

[1]: https://hub.docker.com/r/lmsysorg/sglang/tags "lmsysorg/sglang - Docker Image"
[2]: https://github.com/sgl-project/sglang/issues/20973 "[Bug] can't load AxionML/Qwen3.5-35B-A3B-NVFP4 on fresh `lmsysorg/sglang:dev-cu13` on Nvidia DGX Spark · Issue #20973 · sgl-project/sglang · GitHub"
[3]: https://docs.sglang.ai/advanced_features/server_arguments.html "Server Arguments — SGLang"
