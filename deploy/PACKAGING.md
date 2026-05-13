# Packaging voicepipe — two artifacts

The engine (`pipeline/`) is the same in both. What differs is the shell and the deployment model.

## 1. Native desktop app  (`desktop/`, Tauri)

- A webview + a sidecar (`voicepipe serve`) on an ephemeral **loopback** port.
- **No auth** — the app is trusted by construction; nothing network-facing is meant to be reached
  by anyone else. (Refinement, see `desktop/README.md`: bind a Unix socket and bridge it from the
  Rust side so there's literally no TCP port. TODO.)
- **No systemd** — it starts/stops with the app, killing its sidecar on exit.
- Ships with a **standalone Python** + `voicepipe[gui]` so the user installs nothing. Build the
  Python bundle with PyInstaller or a relocatable `python-build-standalone` runtime; point
  `desktop/src-tauri/src/main.rs::spawn_engine` at it instead of bare `voicepipe` on PATH; list it
  under `bundle.externalBin` in `tauri.conf.json`.
- Does **not** bundle the heavy `[train]` / `[deploy]` extras (torch, bitsandbytes, llama.cpp).
  Training is a CUDA-box job; the app's answer for that is **"connect to a remote engine"** — point
  it at a `voicepipe serve --host 0.0.0.0 --auth-token …` URL (the web deployment, below), which
  it then drives exactly like the local sidecar.

## 2. Web-only deployment  (`deploy/`)

- Just `pip install -e ".[gui]"` (plus `[train]`/`[deploy]` if this box also trains) into a venv,
  + the `deploy/voicepipe.service` systemd unit.
- Bound to a real `--host:--port` with an `--auth-token` (the server **refuses** a non-loopback
  bind without one). The web UI prompts for the token and stores it in `localStorage`.
- **This is the only artifact with a systemd service.** Put it behind a TLS-terminating reverse
  proxy if it's exposed beyond a trusted LAN.
- No GUI app, no bundled Python — it's "the engine, reachable from a browser."

## Auth boundary

Auth (`--auth-token` / `$VOICEPIPE_AUTH_TOKEN`) is enforced **only** on `/v1/*` and **only** when
serving over TCP — it's required automatically for any non-loopback host, optional for
`127.0.0.1`, and **bypassed entirely** when serving over a Unix socket. The native app uses the
loopback/UDS path and never sees an auth prompt; the web deployment always does. Static UI files
are served without auth so the login prompt can load.
