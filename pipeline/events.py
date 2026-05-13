"""
Structured progress events for pipeline stages.

A stage emits newline-delimited JSON events to a sink chosen by environment variable. When no
sink is configured (the normal CLI case) every emit is a no-op, so terminal usage is unchanged.

Sink resolution (checked once, lazily, on the first emit):
  - VOICEPIPE_EVENTS_FD=<n>    write to that already-open file descriptor (a pipe the server passes)
  - VOICEPIPE_EVENTS_FILE=<p>  append to that path
  - (neither)                  no-op
  - VOICEPIPE_EVENTS_STDERR=1  additionally mirror every event to stderr (debugging)

See pipeline/GUI_API.md for the event schema. The contract: `seq` and `job_id` are added by the
consumer (the server), not here; here we add `ts` and `stage`.

Usage in a stage's main():

    from pipeline import events
    events.set_stage("synthesize")
    events.stage_start(command=sys.argv, params={...}, inputs=[...], outputs=[...])
    ...
    events.progress(current=n, total=target, unit="pairs", detail="...")
    events.artifact(path, kind="dataset")
    events.log("something noteworthy", level="warn")
    ...
    events.stage_end(status="ok", summary={...})
"""

import json
import os
import sys
import threading
from datetime import datetime, timezone
from typing import Any, Optional

_lock = threading.Lock()
_stage: Optional[str] = None
_sink = None            # a writable file object, or False once we've determined there is none
_mirror_stderr = False


def set_stage(name: str) -> None:
    """Tag every subsequent event with this stage name. Call once at the top of a stage main()."""
    global _stage
    _stage = name


def _resolve_sink():
    global _sink, _mirror_stderr
    if _sink is not None:
        return _sink
    _mirror_stderr = os.environ.get("VOICEPIPE_EVENTS_STDERR") == "1"
    fd = os.environ.get("VOICEPIPE_EVENTS_FD")
    path = os.environ.get("VOICEPIPE_EVENTS_FILE")
    try:
        if fd is not None and fd.strip() != "":
            _sink = os.fdopen(int(fd), "w", buffering=1, encoding="utf-8", closefd=False)
        elif path:
            _sink = open(path, "a", buffering=1, encoding="utf-8")
        else:
            _sink = False
    except (OSError, ValueError):
        _sink = False
    return _sink


def emit(type: str, **fields: Any) -> None:
    """Emit one event. No-op when no sink is configured."""
    sink = _resolve_sink()
    if sink is False and not _mirror_stderr:
        return
    evt = {"ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
           "stage": _stage, "type": type}
    for k, v in fields.items():
        if v is not None:
            evt[k] = v
    line = json.dumps(evt, ensure_ascii=False, default=str)
    with _lock:
        if sink is not False:
            try:
                sink.write(line + "\n")
            except (OSError, ValueError):
                pass
        if _mirror_stderr:
            print(line, file=sys.stderr, flush=True)


# ---- convenience wrappers (thin; just name the event types) ----

def stage_start(command=None, params=None, inputs=None, outputs=None, resumed_from=None, **extra):
    emit("stage_start", command=command, params=params, inputs=inputs, outputs=outputs,
         resumed_from=resumed_from, **extra)


def phase(name: str, index: int = None, total: int = None, **extra):
    emit("phase", name=name, index=index, total=total, **extra)


def progress(current=None, total=None, unit=None, rate=None, eta_sec=None, detail=None, **extra):
    emit("progress", current=current, total=total, unit=unit, rate=rate, eta_sec=eta_sec,
         detail=detail, **extra)


def metric(**fields):
    emit("metric", **fields)


def artifact(path, kind: str = None, bytes: int = None, **extra):
    emit("artifact", path=str(path), kind=kind, bytes=bytes, **extra)


def log(message: str, level: str = "info", **extra):
    emit("log", level=level, message=message, **extra)


def stage_end(status: str = "ok", exit_code: int = 0, duration_sec: float = None,
              summary=None, error: str = None, **extra):
    emit("stage_end", status=status, exit_code=exit_code, duration_sec=duration_sec,
         summary=summary, error=error, **extra)
