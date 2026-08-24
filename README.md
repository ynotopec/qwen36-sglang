# sglang-qwen36

Serve `nvidia/Qwen3.6-35B-A3B-NVFP4` with SGLang in Docker. An optional,
dependency-free `RadixArk/Qwen3.8-27B-NVFP4` profile is documented in
`.env.example` as a minimal override of the Qwen3.6 defaults.

## Features
- idempotent `install.sh`
- `source run.sh [IP] [PORT]`
- API token required
- compatible H100 / DGX Spark
- optional tools and MTP
- optional Qwen3.8 NVFP4 configuration using the packages from the base image

## Requirements
- Docker
- NVIDIA Container Toolkit
- NVIDIA GPU
- enough VRAM / memory headroom

## Install
```bash
./install.sh
```

If `.env` does not exist, it is created from `.env.example`.

`install.sh` and `run.sh` now treat `.env.example` as the default source of truth: any unset variable is loaded from commented defaults in `.env.example`, and `.env` only overrides what you change. This includes `BASE_IMAGE`, so you can pin the upstream SGLang Docker image/tag in `.env` after copying it from `.env.example`.

## Configure

Edit `.env` and set at least:

```bash
API_KEY=replace-with-a-private-token
```

Quote values containing shell metacharacters, for example
`API_KEY='a-private$key'`, because `.env` is sourced by Bash.

Optional:

* `HF_TOKEN` if the model download requires authentication
* `ADMIN_API_KEY` for admin endpoints; when unset it defaults to `API_KEY`
* `ENABLE_MTP=0` to disable speculative decoding / MTP (`1` by default)
* `SPECULATIVE_ALGORITHM=EAGLE` to select EAGLE instead of the default `NEXTN`
* `MAMBA_RADIX_CACHE_STRATEGY=extra_buffer` to tune MTP mamba radix-cache scheduling without using the deprecated SGLang scheduler flag
* `MOE_RUNNER_BACKEND=flashinfer_cutlass` selects the NVFP4-compatible FlashInfer MoE backend (the wrapper default); override it only when your model and SGLang build support another backend
* `CHUNKED_PREFILL_SIZE=4096` to pass `--chunked-prefill-size`; leave it unset to omit the SGLang flag
* `MAX_PREFILL_TOKENS=8192` to pass `--max-prefill-tokens`; leave it unset to omit the SGLang flag
* `ATTENTION_BACKEND=flashinfer`, `DISABLE_PREFILL_CUDA_GRAPH=1`, and `MAMBA_FULL_MEMORY_RATIO=4.59` expose the remaining Qwen3.8 launch settings
* `ENABLE_MIXED_CHUNK=1` to enable SGLang mixed-chunk scheduling (`0` by default)
* `ENABLE_SLEEP_ON_IDLE=0` to opt out of SGLang `--sleep-on-idle` (`1` by default)
* `TOOL_SERVER=...` if using tool execution
* `KV_CACHE_DTYPE=...` to override KV cache precision (for example `auto`, `fp8_e4m3`, or `fp8_e5m2`)
* `FLASHINFER_DISABLE_VERSION_CHECK=0` if you want to re-enable strict `flashinfer`/`flashinfer-jit-cache` version checks
* `BASE_IMAGE=lmsysorg/sglang:latest-runtime` to use the default stable runtime image, or set a versioned tag to pin the upstream SGLang release used by `install.sh`
* `UPGRADE_SGLANG=1` only if you explicitly want to replace base-image SGLang binaries (off by default for CUDA compatibility)
* `UPGRADE_TRANSFORMERS=1` only if you explicitly want a bleeding-edge `transformers` build

If you omit `HF_TOKEN`, Hugging Face may repeatedly print unauthenticated/rate-limit warnings during model download.

Hub downloads are stored under `${HOME}/.cache/huggingface/hub` and mounted as
`/app/models/hub` in the container.

To reproduce the RadixArk Qwen3.8 DFlash command entirely from `.env`, copy its
profile from the end of `.env.example`, set `API_KEY`, then run `./run.sh` with
no positional arguments:

```dotenv
API_KEY=replace-with-a-private-token
MODEL_PATH=RadixArk/Qwen3.8-27B-NVFP4
SERVED_MODEL_NAME=qwen3.8
HTTP_PORT=30000
MEM_FRACTION_STATIC=0.80
TRUST_REMOTE_CODE=1
ATTENTION_BACKEND=flashinfer
CHUNKED_PREFILL_SIZE=2048
MAMBA_FULL_MEMORY_RATIO=11.01
MAMBA_RADIX_CACHE_STRATEGY=extra_buffer_lazy
MAMBA_SSM_DTYPE=float32
ENABLE_SLEEP_ON_IDLE=0
SPECULATIVE_ALGORITHM=DFLASH
SPECULATIVE_DRAFT_MODEL_PATH=incoai/Qwen3.8-27B-DFlash2
SPECULATIVE_NUM_DRAFT_TOKENS=8
```

The spaced comments in the optional profile are deliberately excluded from the
default loader. The Qwen3.6 defaults already provide FP8 KV cache, MTP, and the
reasoning/tool parsers, so they are intentionally absent from this minimal
override. `ENABLE_SLEEP_ON_IDLE=0` removes the Qwen3.6 default flag that was not
present in the supplied Qwen3.8 command.
Positional arguments to `run.sh`, when supplied, still override `HOST` and
`PUBLISH_PORT`.

To reproduce the NVIDIA Qwen3.6 NVFP4/EAGLE launch profile with the pinned
`v0.5.15.post1-cu130` base image, copy the matching minimal block from
`.env.example` into `.env`, set `API_KEY`, then run `./install.sh` and
`./run.sh 0.0.0.0 8572`. Values omitted from that block already match the
wrapper defaults. The wrapper additionally enables API authentication and uses
its managed-container defaults (named container, restart policy, cache mount,
and port publishing) rather than the one-shot `docker run` lifecycle.

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

* SGLang supports `--api-key`, `--admin-api-key`, `--reasoning-parser`, `--tool-call-parser`, and speculative decoding flags.
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
