"""Frozen entry point for the bundled voicepipe engine (the Tauri desktop sidecar).
Built with PyInstaller (see ../build-sidecar.sh) into a --onedir bundle that ships inside the
.app's Resources/. It's just the `voicepipe` CLI — the shell invokes it as
`voicepipe-serve serve --host 127.0.0.1 --port <p> --no-browser --no-auth`."""
import multiprocessing
multiprocessing.freeze_support()   # required first for frozen apps (no-op when not frozen)

import sys
from pipeline.cli import main

if __name__ == "__main__":
    sys.exit(main())
