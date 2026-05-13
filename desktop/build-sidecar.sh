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

# --strip removes debug symbols from the embedded interpreter + .so/.dylib files; on macOS/Linux
# this is a non-trivial size win for the shipped bundle. Skip on Windows where it's a no-op /
# can fail without the GNU binutils strip in PATH.
case "$(uname -s 2>/dev/null || true)" in
  MINGW*|MSYS*|CYGWIN*) STRIP_FLAG="" ;;
  *)                    STRIP_FLAG="--strip" ;;
esac

# shellcheck disable=SC2086  # STRIP_FLAG is a single optional arg or empty; intentional word-split
"$PYI" --noconfirm --clean --onedir --name voicepipe-serve $STRIP_FLAG \
  --distpath "$HERE/sidecar/dist" --workpath "$HERE/sidecar/build" --specpath "$HERE/sidecar" \
  --paths "$REPO" \
  --collect-all pipeline \
  --collect-submodules uvicorn \
  --hidden-import uvicorn.lifespan.on --hidden-import uvicorn.lifespan.off \
  --hidden-import uvicorn.protocols.http.h11_impl --hidden-import uvicorn.loops.asyncio --hidden-import h11 \
  --exclude-module torch --exclude-module transformers --exclude-module peft --exclude-module trl \
  --exclude-module bitsandbytes --exclude-module datasets --exclude-module accelerate \
  "$HERE/sidecar/voicepipe_sidecar.py"

# NOTE: Tauri's `bundle.externalBin` is the proper pattern for sidecar binaries (it does the
# platform-triple renaming + signing on macOS/Windows). We DEFER it here because PyInstaller's
# --onedir output is a directory, not a single binary, and externalBin doesn't natively wrap
# directories. Switching to --onefile would resolve that but inflates cold-start by ~1-2s and
# changes the resource-extract model — out of scope for this round. tauri.conf.json therefore
# still ships the bundle via bundle.resources; revisit when codesigning / notarization is wired.
echo "built: $HERE/sidecar/dist/voicepipe-serve/voicepipe-serve  ($(du -sh "$HERE/sidecar/dist/voicepipe-serve" | cut -f1))"
