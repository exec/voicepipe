# voicepipe-worker — Docker image for remote workers

A self-contained Docker image that runs `voicepipe serve` on a public port
with a **per-boot random auth token**, suitable as a vast.ai / runpod /
hetzner template (or any host where you can `docker run`). The user's local
voicepipe GUI pairs with it via the existing "Add remote engine" panel.

Released to **`ghcr.io/exec/voicepipe-worker:latest`** on every `v*` tag via
`.github/workflows/release.yml`. No Docker Hub mirror.

## How pairing works (no shared secrets)

On startup, the container:

1. Generates a random URL-safe 32-byte token (`python3 -c 'secrets.token_urlsafe(32)'`)
2. Prints a clearly-fenced pairing block to **stdout**
3. Writes the same block to `/workspace/voicepipe-pairing.txt` inside the container
4. Execs `voicepipe serve --host 0.0.0.0 --auth-token $TOK --no-browser`

The token never leaves the container except via container logs. The user reads
the block from their cloud provider's web console (vast.ai, runpod, etc. all
show container stdout) and pastes the token + the provider's forwarded
host:port into the GUI's "Add remote engine" field. From there the GUI hits
the worker over plain HTTP with `Authorization: Bearer <token>` — every stage
runs remotely as if it were local.

## Run

GPU box (vast.ai / runpod / hetzner with NVIDIA + NVIDIA Container Runtime):

```
docker run --gpus all -p 8765:8765 -d ghcr.io/exec/voicepipe-worker:latest
docker logs -f <container-id>   # read the pairing block
```

Local test without GPU (the engine starts; train/deploy stages will fail
when invoked but the pairing flow itself works):

```
docker run --rm -p 8765:8765 ghcr.io/exec/voicepipe-worker:latest
curl -H "Authorization: Bearer <token>" http://localhost:8765/v1/health
```

## Env vars (all optional)

| | |
|---|---|
| `VOICEPIPE_AUTH_TOKEN` | Override the auto-generated token. Must be ≥32 chars / ≥2 char classes / ≥8 distinct chars (the server enforces this for non-loopback binds). |
| `VOICEPIPE_PORT` | Default 8765. If overridden, also `EXPOSE` and `-p` the new port. |
| `HF_HUB_ENABLE_HF_TRANSFER` | Defaults to `1`. `hf_transfer` accelerates HuggingFace model downloads ~10-30x; turn off only if it misbehaves. |

The token is generated **inside** the container — there is no shared secret
between the image and any external system. Same image runs anywhere, fresh
token every boot.

## Build locally

```
./deploy/docker/build-worker.sh                                  # → ghcr.io/exec/voicepipe-worker:latest
TAG=ghcr.io/exec/voicepipe-worker:dev ./deploy/docker/build-worker.sh  # custom tag
```

Build context is the repo root (the script handles the `cd`). Excludes are in
the top-level `.dockerignore` — only `pyproject.toml`, `constraints-train.txt`,
and `pipeline/` enter the image. Datasets / scratch / desktop / virtualenvs
never get baked in.

## Hardware

Pre-built image targets **Blackwell (sm_120, RTX 50-series)** with `torch
2.11.0` from PyTorch's `cu128` index. For Ada/Ampere (sm_86/89, RTX 30/40),
edit the Dockerfile's `pip install ... torch==...` line to `torch==2.5.1`
from `.../cu121` — the rest of the pins in `constraints-train.txt` are
CUDA-agnostic. See the Ada/Ampere fallback block at the top of that file.

## Size

~8 GB compressed. The image bundles torch + transformers + peft + trl +
bitsandbytes + datasets + llama.cpp — every dependency for every stage. This
is intentional: one image runs synth, dedup, triage, assemble, train, deploy.
The cost is bandwidth on first pull; thereafter image layers are cached.

## Security notes

- **HTTP, not HTTPS.** The token transits cleartext over the public internet
  on every API call. The 32-byte token is high-entropy so guessing isn't
  practical, but any network observer on the path can capture it. Comparable
  risk profile to `ssh root@cloudbox` without host-key verification.
- **Per-boot token rotation.** Destroying and recreating the container gives a
  new token. There is no out-of-band recovery — if you lose the pairing block,
  destroy and re-pair.
- **Worker is bound to 0.0.0.0.** It accepts connections from anywhere that
  can reach the forwarded port. The token is the only gate. Set firewall
  rules at the provider level if you want narrower access.

For TLS / NAT traversal, you can front the container with `cloudflared`,
`traefik`, `caddy`, or join it to a Tailscale net. Not bundled — keep the
default image dependency-free.

## CI

`.github/workflows/release.yml` has a `worker-image` job that runs on every
`v*` tag push. It builds with `docker/build-push-action` and pushes
`ghcr.io/exec/voicepipe-worker:<tag>` and `:latest` using the workflow's
`GITHUB_TOKEN` for ghcr auth. No registry credentials need to be set as
repo secrets.
