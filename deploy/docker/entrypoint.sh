#!/bin/sh
# voicepipe-worker entrypoint: generate a per-boot auth token, print a pairing
# block to stdout (so the user can read it from the cloud provider's web
# console), then exec `voicepipe serve` bound on 0.0.0.0.
#
# Token is generated INSIDE the container. No shared secret is required between
# the image and any external system: the user reads it from container logs and
# pastes it into the local voicepipe GUI's "Add remote engine" panel.
#
# Optional env vars (all opt-in):
#   VOICEPIPE_AUTH_TOKEN — override the auto-generated token (e.g. for a long-
#                          running worker the user already has paired). Must be
#                          ≥32 chars / ≥2 char classes / ≥8 distinct chars.
#   VOICEPIPE_PORT       — override the default 8765 (the container exposes 8765
#                          by default; if you change this, expose the new port).

set -eu

PORT="${VOICEPIPE_PORT:-8765}"
TOKEN="${VOICEPIPE_AUTH_TOKEN:-$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')}"

# Best-effort external-IP probe. On vast.ai / runpod the egress IP often
# differs from the inbound-NAT IP; the user should check their provider's
# "open ports" panel for the real public address. We print it as a hint, not
# as authoritative.
EXTERNAL_IP="$(curl -s --max-time 5 https://api.ipify.org 2>/dev/null || echo 'unknown')"

PAIRING_FILE=/workspace/voicepipe-pairing.txt

cat <<EOF | tee "${PAIRING_FILE}"

================================================================
                  VOICEPIPE WORKER — PAIRING
================================================================

  Token:  ${TOKEN}

  Internal endpoint: 0.0.0.0:${PORT}
  External (egress, best guess): ${EXTERNAL_IP}:${PORT}

  vast.ai / runpod port-forward to a DIFFERENT external port — check
  the instance's "Open Ports" panel for the actual public host:port.

  In your local voicepipe GUI, click "Add remote engine" and paste:
      Endpoint: <the public host:port from above>
      Token:    ${TOKEN}

  (This block is also saved at ${PAIRING_FILE} inside the container.)

================================================================

EOF

exec voicepipe serve \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --auth-token "${TOKEN}" \
    --no-browser
