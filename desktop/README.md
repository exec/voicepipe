# voicepipe desktop (Tauri)

A thin native shell around the voicepipe control server + web UI. The window is a webview; a
sidecar process (`voicepipe serve`) runs the Python engine; the webview talks to it over
loopback. The native app is trusted by construction — **no auth**. (Auth + `--host 0.0.0.0`
belong to the *web-only* deployment, which is the one that gets a systemd service — see
`../deploy/`.)

Why Tauri (Rust) over Wails (Go): the backend language does almost nothing here — spawn and
supervise the sidecar, native dialogs, the window. Tauri's sidecar + bundler story is the better
fit for shipping a Python runtime alongside, and the binaries are small. If you'd rather work in
Go, Wails does the same job; the web UI under `../pipeline/webui/` is unchanged either way.

## Dev

```bash
# 1. have the engine available on PATH (so `voicepipe serve` works):
pip install -e ".[gui]"            # from the repo root
# 2. run the shell against the dev server:
cd desktop && cargo tauri dev      # needs the Rust toolchain + `cargo install tauri-cli`
```

`src-tauri/src/main.rs` spawns `voicepipe serve --port <free-loopback-port> --no-browser` on
startup, waits for `/v1/health`, then points the window at it; it kills the sidecar on exit.

## Building

```bash
desktop/build-sidecar.sh          # builds the self-contained engine -> desktop/sidecar/dist/voicepipe-serve/ (needs: .venv/bin/pip install pyinstaller)
cd desktop && cargo tauri build   # compiles the shell + copies the sidecar into the .app's Resources/ -> .app + .dmg
```

The `.app` is **self-contained** — it ships a PyInstaller `--onedir` bundle of `voicepipe[gui]`
(CPython + fastapi/uvicorn/requests/numpy + the `pipeline` package + `webui/` + `templates/`) at
`Contents/Resources/voicepipe-serve/`, and `main.rs::locate_engine` runs that first (it falls
back to a PATH `voicepipe` / a dev `.venv` for `cargo tauri dev`). The heavy `[train]`/`[deploy]`
extras (torch, bitsandbytes, llama.cpp) are **not** bundled — training is a CUDA-box job; the
app's answer for that is "connect to a remote engine" (point at a `voicepipe serve --host
0.0.0.0 --auth-token …` URL — see `../deploy/`). ~26 MB `.dmg`.

## Packaging — two artifacts (see `../deploy/PACKAGING.md`)

1. **Native app** (this directory): the self-contained `.app`/`.dmg` built above. Still **ad-hoc
   signed** — for distribution you need an Apple Developer ID + notarization
   (`tauri.conf.json` `bundle.macOS.signingIdentity` + the notarization env vars; Tauri's bundler
   does the `codesign`/`notarytool`/`stapler` dance). Without it, Gatekeeper requires
   right-click→Open (or `xattr -dr com.apple.quarantine`).
2. **Web-only** (`../deploy/`): just `voicepipe[gui]` + a systemd unit, bound to a host:port with
   an auth token. No GUI app, no bundled Python — a server you reach over a browser.

## Open TODOs

- **The "no TCP port at all" refinement**: the sidecar still binds an ephemeral loopback port.
  To bind *nothing* network-facing: run `voicepipe serve --unix-socket <dir>/voicepipe.sock` and
  bridge the webview's HTTP calls to that UDS (a tiny in-process Rust proxy, or Tauri custom
  protocol + command handlers). `voicepipe serve --unix-socket` already works; the Rust bridge is unbuilt.
- **Windows / Linux**: the shell works there with the normal frame, but needs build runners, an
  MSI/NSIS installer (signed) for Windows, `.deb`/`.AppImage` for Linux, a Windows/Linux build of
  the PyInstaller sidecar, and `decorations: false` + custom min/max/close buttons in the webui
  to drop the title bar there (no overlay-traffic-lights equivalent off macOS).
- **CI + auto-updater**: Tauri's GitHub Action builds/signs/notarizes on tags; the updater plugin
  needs a signed manifest + a hosting endpoint.
