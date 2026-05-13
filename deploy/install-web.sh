#!/usr/bin/env bash
# Set up the WEB-ONLY voicepipe deployment on a Linux box (the one with the systemd service).
# Run as root. Idempotent-ish. Adjust HOST/PORT/PREFIX below if you like.
set -euo pipefail

PREFIX="${PREFIX:-/opt/voicepipe}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8765}"
SVC_USER="${SVC_USER:-voicepipe}"
REPO_SRC="${1:-$(cd "$(dirname "$0")/.." && pwd)}"   # path to a checkout of this repo

id -u "$SVC_USER" >/dev/null 2>&1 || useradd --system --create-home --home-dir "$PREFIX" --shell /usr/sbin/nologin "$SVC_USER"
mkdir -p "$PREFIX/projects" /etc/voicepipe
[ -d "$PREFIX/src" ] || cp -r "$REPO_SRC" "$PREFIX/src"
python3 -m venv "$PREFIX/.venv"
"$PREFIX/.venv/bin/pip" install --upgrade pip
"$PREFIX/.venv/bin/pip" install -e "$PREFIX/src[gui]"
# add [train],[deploy] here too if this box also fine-tunes:  ...[gui,train,deploy]

if [ ! -f /etc/voicepipe/env ]; then
  TOKEN="$(openssl rand -hex 24)"
  printf 'VOICEPIPE_AUTH_TOKEN=%s\n' "$TOKEN" > /etc/voicepipe/env
  chmod 600 /etc/voicepipe/env
  echo "generated auth token: $TOKEN  (also in /etc/voicepipe/env)"
fi
chown -R "$SVC_USER:$SVC_USER" "$PREFIX" /etc/voicepipe

# write the unit, substituting host/port/prefix
sed -e "s#/opt/voicepipe#$PREFIX#g" \
    -e "s#--host 0.0.0.0#--host $HOST#" \
    -e "s#--port 8765#--port $PORT#" \
    -e "s#^User=voicepipe#User=$SVC_USER#" -e "s#^Group=voicepipe#Group=$SVC_USER#" \
    "$REPO_SRC/deploy/voicepipe.service" > /etc/systemd/system/voicepipe.service
systemctl daemon-reload
systemctl enable --now voicepipe
echo "voicepipe is up on http://$HOST:$PORT  — use the token from /etc/voicepipe/env to log in."
systemctl --no-pager status voicepipe || true
