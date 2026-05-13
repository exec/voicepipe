#!/usr/bin/env bash
# Set up the WEB-ONLY voicepipe deployment on a Linux box (the one with the systemd service).
# Run as root. Idempotent — re-running upgrades the code tree and unit without clobbering state.
#
# Layout:
#   /opt/voicepipe/src      → repo source, read-only after install (owned by root, mode 0755)
#   /opt/voicepipe/.venv    → venv with voicepipe[gui] installed
#   /var/lib/voicepipe      → projects/ + per-user state (HF/Ollama caches, jobs/, …). The service
#                              user owns this; systemd ProtectHome=true is happy because the user's
#                              home is /nonexistent, not /opt/voicepipe.
#   /etc/voicepipe/env      → mode 0600, holds VOICEPIPE_AUTH_TOKEN
set -euo pipefail

PREFIX="${PREFIX:-/opt/voicepipe}"
STATE_DIR="${STATE_DIR:-/var/lib/voicepipe}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8765}"
SVC_USER="${SVC_USER:-voicepipe}"
REPO_SRC="${1:-$(cd "$(dirname "$0")/.." && pwd)}"   # path to a checkout of this repo

id -u "$SVC_USER" >/dev/null 2>&1 || useradd --system --home-dir /nonexistent --shell /usr/sbin/nologin "$SVC_USER"
mkdir -p "$PREFIX" "$STATE_DIR/projects" /etc/voicepipe

# Copy the repo source into $PREFIX/src, excluding VCS/venv/secrets/build/scratch detritus. Prefer
# rsync (handles deletes + has --exclude); otherwise fall back to `find … -print0 | cpio -pdm`
# with the same exclusions. The result is a clean source tree owned by root, root.
EXCLUDES=(
    ".git" ".venv" "venv" "scratch"
    ".env" "*.env"
    "__pycache__" "*.pyc" "*.pyo"
    "*.egg-info" "dist" "build" "target"
    "node_modules"
    "desktop/sidecar/dist" "desktop/sidecar/build"
    ".DS_Store"
)
mkdir -p "$PREFIX/src"
if command -v rsync >/dev/null 2>&1; then
    RSYNC_ARGS=(-a --delete)
    for e in "${EXCLUDES[@]}"; do
        # keep .env.example even though we exclude *.env
        if [ "$e" = "*.env" ]; then
            RSYNC_ARGS+=(--include=".env.example" --exclude="$e")
        else
            RSYNC_ARGS+=(--exclude="$e")
        fi
    done
    rsync "${RSYNC_ARGS[@]}" "$REPO_SRC/" "$PREFIX/src/"
else
    # `find … -print0 | cpio -pdm` preserves dirs; build a -prune expression from EXCLUDES.
    FIND_ARGS=("$REPO_SRC")
    FIRST=1
    for e in "${EXCLUDES[@]}"; do
        [ "$e" = "*.env" ] && continue   # rsync-only; cpio path keeps `.env` files but they're caught by `.env` literal above
        if [ "$FIRST" = "1" ]; then
            FIND_ARGS+=(\( -name "$e")
            FIRST=0
        else
            FIND_ARGS+=(-o -name "$e")
        fi
    done
    FIND_ARGS+=(\) -prune -o -print0)
    (cd "$REPO_SRC" && find "${FIND_ARGS[@]}" | cpio -pdm0 --quiet "$PREFIX/src")
fi
chown -R root:root "$PREFIX/src"
chmod -R a-w,a+rX "$PREFIX/src"     # read-only source tree after install

# venv lives under $PREFIX (writable by root only; the service user just reads from it)
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
chown -R "$SVC_USER:$SVC_USER" "$STATE_DIR"
chown root:"$SVC_USER" /etc/voicepipe/env
chmod 640 /etc/voicepipe/env

# write the unit, substituting host/port/prefix/state-dir/user
sed -e "s#@PREFIX@#$PREFIX#g" \
    -e "s#@STATE_DIR@#$STATE_DIR#g" \
    -e "s#@HOST@#$HOST#g" \
    -e "s#@PORT@#$PORT#g" \
    -e "s#@SVC_USER@#$SVC_USER#g" \
    "$REPO_SRC/deploy/voicepipe.service" > /etc/systemd/system/voicepipe.service
systemctl daemon-reload
systemctl enable --now voicepipe
echo "voicepipe is up on http://$HOST:$PORT  — use the token from /etc/voicepipe/env to log in."
systemctl --no-pager status voicepipe || true
