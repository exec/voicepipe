#!/usr/bin/env bash
# Build the self-contained `voicepipe-serve` engine bundle that ships inside the desktop app.
# Run from anywhere. By default uses the repo's .venv; set $VENV to point at another env (CI does).
# Output: desktop/sidecar/dist/voicepipe-serve/  (a --onedir PyInstaller bundle, .exe on Windows).
# The Tauri build copies that folder into the app's Resources/ (see tauri.conf.json).
#
# Requires:  pyinstaller in the env  (`pip install pyinstaller`) — engine deps come from `pip install -e .[gui]`.
# Does NOT bundle the [train]/[deploy] extras (torch, llama.cpp) — those run on a GPU box / via a remote engine.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
VENV="${VENV:-$REPO/.venv}"
PYI=""
for cand in "$VENV/bin/pyinstaller" "$VENV/Scripts/pyinstaller.exe" "$VENV/Scripts/pyinstaller"; do
  [ -x "$cand" ] && { PYI="$cand"; break; }
done
[ -n "$PYI" ] || PYI="$(command -v pyinstaller || true)"
[ -n "$PYI" ] || { echo "pyinstaller not found (looked in $VENV/{bin,Scripts}/ and PATH) — run: pip install pyinstaller"; exit 1; }

rm -rf "$HERE/sidecar/dist" "$HERE/sidecar/build" "$HERE/sidecar/voicepipe-serve.spec"
"$PYI" --noconfirm --clean --onedir --name voicepipe-serve \
  --distpath "$HERE/sidecar/dist" --workpath "$HERE/sidecar/build" --specpath "$HERE/sidecar" \
  --paths "$REPO" \
  --collect-all pipeline \
  --collect-submodules uvicorn \
  --hidden-import uvicorn.lifespan.on --hidden-import uvicorn.lifespan.off \
  --hidden-import uvicorn.protocols.http.h11_impl --hidden-import uvicorn.loops.asyncio --hidden-import h11 \
  --exclude-module torch --exclude-module transformers --exclude-module peft --exclude-module trl \
  --exclude-module bitsandbytes --exclude-module datasets --exclude-module accelerate \
  "$HERE/sidecar/voicepipe_sidecar.py"
echo "built: $HERE/sidecar/dist/voicepipe-serve/voicepipe-serve  ($(du -sh "$HERE/sidecar/dist/voicepipe-serve" | cut -f1))"
