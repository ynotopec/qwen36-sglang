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

````

---

## systemd user service
```ini
[Unit]
Description=SGLang Qwen3.6 35B A3B FP8
After=docker.service
Requires=docker.service

[Service]
Type=simple
WorkingDirectory=/home/ailab/sglang-qwen36
ExecStart=/usr/bin/bash -lc 'source /home/ailab/sglang-qwen36/run.sh 0.0.0.0 8080'
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
````

---

## réglages que je te conseille

Pour **H100** :

```bash
MEM_FRACTION_STATIC=0.80
CONTEXT_LENGTH=131072
MAX_RUNNING_REQUESTS=32
ENABLE_MTP=0
```

Pour **DGX Spark**, pars plus bas, car il y a déjà eu des bugs signalés sur `dev-cu13` / DGX Spark avec certains modèles quantisés, donc je préfère te donner une base prudente plutôt qu’un réglage trop agressif. ([GitHub][2])

```bash
MEM_FRACTION_STATIC=0.72
CONTEXT_LENGTH=65536
MAX_RUNNING_REQUESTS=8
ENABLE_MTP=0
```

Le point important : SGLang recommande de **baisser `--mem-fraction-static`** en cas d’OOM, et de **réduire le chunked prefill size** pour les longs prompts. ([docs.sglang.ai][3])

## usage

```bash
cd /home/ailab/sglang-qwen36
./install.sh
cp -n .env.example .env
nano .env
source ./run.sh 0.0.0.0 8080
```

## test

```bash
export API_KEY='change-me-very-strong-token'

curl http://127.0.0.1:8080/v1/models \
  -H "Authorization: Bearer ${API_KEY}"

curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d '{
    "model": "qwen3.6-35b-a3b-fp8",
    "messages": [{"role": "user", "content": "Salut"}]
  }'
```

[1]: https://hub.docker.com/r/lmsysorg/sglang/tags "lmsysorg/sglang - Docker Image"
[2]: https://github.com/sgl-project/sglang/issues/20973 "[Bug] can't load AxionML/Qwen3.5-35B-A3B-NVFP4 on fresh `lmsysorg/sglang:dev-cu13` on Nvidia DGX Spark · Issue #20973 · sgl-project/sglang · GitHub"
[3]: https://docs.sglang.ai/advanced_features/server_arguments.html "Server Arguments — SGLang"
