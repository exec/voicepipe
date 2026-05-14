#!/usr/bin/env bash
# Build the voicepipe-worker docker image from the repo root.
#
# Override the tag via env: TAG=ghcr.io/exec/voicepipe-worker:v0.2 ./build-worker.sh
set -euo pipefail

cd "$(dirname "$0")/../.."  # repo root
TAG="${TAG:-ghcr.io/exec/voicepipe-worker:latest}"

echo "Building $TAG from $(pwd)"
docker build -f deploy/docker/Dockerfile -t "$TAG" .

echo
echo "Built: $TAG"
docker images "$TAG" --format 'table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}'
echo
echo "Push:  docker push $TAG"
echo "Run:   docker run --gpus all -p 8765:8765 -d $TAG"
echo "Logs:  docker logs -f \$(docker ps -q --filter ancestor=$TAG | head -1)"
