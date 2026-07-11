#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="$(basename "${PROJECT_DIR}")"
cd "${PROJECT_DIR}"

command -v docker >/dev/null 2>&1 || {
  echo "Missing dependency: docker" >&2
  exit 1
}

if [[ ! -f .env && -f .env.example ]]; then
  cp .env.example .env
  echo "Created .env from .env.example"
fi

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

load_example_defaults() {
  local example_file="${PROJECT_DIR}/.env.example"
  [[ -f "${example_file}" ]] || return 0

  while IFS= read -r line; do
    [[ "${line}" =~ ^#[A-Z0-9_]+= ]] || continue

    local kv="${line#\#}"
    local var_name="${kv%%=*}"
    local raw_value="${kv#*=}"

    if [[ -z "${!var_name+x}" || -z "${!var_name}" ]]; then
      local expanded
      expanded=$(eval "echo \"${raw_value}\"")
      export "${var_name}=${expanded}"
    fi
  done < "${example_file}"
}

load_example_defaults

IMAGE_NAME="${IMAGE_NAME:-${PROJECT_NAME}:latest}"
BASE_IMAGE="${BASE_IMAGE:-lmsysorg/sglang:dev-cu13}"
UPGRADE_TRANSFORMERS="${UPGRADE_TRANSFORMERS:-0}"
UPGRADE_SGLANG="${UPGRADE_SGLANG:-0}"
SGLANG_WHL_INDEX="${SGLANG_WHL_INDEX:-https://docs.sglang.ai/whl/cu130/}"

chmod +x entrypoint.sh run.sh || true

echo "Pulling base image: ${BASE_IMAGE}"
docker pull "${BASE_IMAGE}"

echo "Building ${IMAGE_NAME}..."
docker build \
  --pull \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  --build-arg UPGRADE_TRANSFORMERS="${UPGRADE_TRANSFORMERS}" \
  --build-arg UPGRADE_SGLANG="${UPGRADE_SGLANG}" \
  --build-arg SGLANG_WHL_INDEX="${SGLANG_WHL_INDEX}" \
  -t "${IMAGE_NAME}" \
  .

echo "Built ${IMAGE_NAME}"
echo "Run with:"
echo "  source ./run.sh 0.0.0.0 8080"
