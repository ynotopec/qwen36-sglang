#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="$(basename "${PROJECT_DIR}")"
IMAGE_NAME="${IMAGE_NAME:-${PROJECT_NAME}:latest}"
BASE_IMAGE="${BASE_IMAGE:-lmsysorg/sglang:dev-cu13}"

cd "${PROJECT_DIR}"

command -v docker >/dev/null 2>&1 || {
  echo "Missing dependency: docker" >&2
  exit 1
}

if [[ ! -f .env && -f .env.example ]]; then
  cp .env.example .env
  echo "Created .env from .env.example"
fi

chmod +x entrypoint.sh run.sh || true

echo "Pulling base image: ${BASE_IMAGE}"
docker pull "${BASE_IMAGE}"

echo "Building ${IMAGE_NAME}..."
docker build \
  --pull \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  -t "${IMAGE_NAME}" \
  .

echo "Built ${IMAGE_NAME}"
echo "Run with:"
echo "  source ./run.sh 0.0.0.0 8080"
