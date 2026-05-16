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

IMAGE_NAME="${IMAGE_NAME:-${PROJECT_NAME}:latest}"
BASE_IMAGE="${BASE_IMAGE:-lmsysorg/sglang:dev-cu13}"
UPGRADE_TRANSFORMERS="${UPGRADE_TRANSFORMERS:-0}"
UPGRADE_SGLANG="${UPGRADE_SGLANG:-0}"
SGLANG_WHL_INDEX="${SGLANG_WHL_INDEX:-https://docs.sglang.ai/whl/cu130/}"
SGLANG_GIT_REF="${SGLANG_GIT_REF:-refs/pull/20547/head}"
HUGGINGFACE_HUB_SPEC="${HUGGINGFACE_HUB_SPEC:->=0.36.0,<1.0}"

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
  --build-arg SGLANG_GIT_REF="${SGLANG_GIT_REF}" \
  --build-arg HUGGINGFACE_HUB_SPEC="${HUGGINGFACE_HUB_SPEC}" \
  -t "${IMAGE_NAME}" \
  .

echo "Built ${IMAGE_NAME}"
echo "Run with:"
echo "  source ./run.sh 0.0.0.0 8080"
