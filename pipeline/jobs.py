"""
Job/process supervision for pipeline stages.

A *job* is one run of one stage (`python -m pipeline.<stage> --project DIR ...`), launched as a
detached process group with its event stream pointed at a file. Everything about it is persisted
under `<project>/dataset/jobs/<job_id>/`:

    meta.json        — pid, stage, project, status, started_at, ended_at, exit_code, command
    events.ndjson    — the structured event stream (from pipeline.events; the line number is `seq`)
    console.log      — captured stdout+stderr (the human text the stage prints)

This module has no web dependency; `pipeline.server` wraps it. It also has no `pipeline.train`
import-time cost — stages are run as subprocesses, never imported here.
"""

# Lazy annotations: `JobManager.list` (the public listing method) shadows the builtin `list`
# inside the class body, which would otherwise make later annotations like `list[str]` resolve
# to the method object at class-definition time → `TypeError: 'function' object is not
# subscriptable`. PEP 563 makes annotations strings, so the shadow can't fire.
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

_STAGES = {"categorize", "synthesize", "dedup", "triage", "assemble", "train", "deploy", "eval", "infer"}
# Stages that write into the same place — don't run two of the same for one project at once.
_EXCLUSIVE = _STAGES

# Override-key allow-list: stages take flags named like `--max-seq-len`. We accept the same shape
# (lowercase, digits, hyphens; must start with a letter) and reject anything else outright — no
# leading dashes, no equals signs, no spaces. Popen is list-based so this isn't shell injection;
# it's argv injection (e.g. an `--exec-foo` flag that doesn't exist for this stage but does for
# some future one) and weird argparse-eating-the-next-token edge cases.
_OVERRIDE_KEY_RE = re.compile(r"[a-z][a-z0-9_-]*")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _new_job_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + os.urandom(3).hex()


def _pid_alive(pid: int) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _proc_start_time(pid: int) -> float | None:
    """Best-effort wall-clock start time for `pid` (epoch seconds). Used to detect PID recycling
    after a server restart. Linux: read /proc/{pid}/stat field 22 (clock ticks since boot, +
    btime). macOS: `ps -o lstart= -p {pid}`. Returns None on failure."""
    if not pid:
        return None
    proc_stat = Path(f"/proc/{pid}/stat")
    if proc_stat.is_file():
        try:
            txt = proc_stat.read_text()
            # field 2 is "(comm)" which may contain spaces — start parsing after the last ')'
            after = txt.rsplit(")", 1)[1].split()
            starttime_ticks = int(after[19])  # field 22 overall, index 19 after the rsplit
            try:
                ticks_per_sec = os.sysconf("SC_CLK_TCK") or 100
            except (ValueError, OSError):
                ticks_per_sec = 100
            with open("/proc/stat") as f:
                btime = next((int(line.split()[1]) for line in f if line.startswith("btime")), None)
            if btime is not None:
                return btime + starttime_ticks / ticks_per_sec
        except (OSError, ValueError, IndexError):
            return None
    try:
        out = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)],
                             capture_output=True, text=True, timeout=2)
        s = (out.stdout or "").strip()
        if not s:
            return None
        import time as _t
        return _t.mktime(_t.strptime(s, "%a %b %d %H:%M:%S %Y"))
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _flags_from_overrides(overrides: dict | None) -> list[str]:
    """{"max_seq_len": 4096, "no_downsample": true, "model": "x"} -> ["--max-seq-len","4096","--no-downsample","--model","x"]
    Keys are restricted to lowercase-letters/digits/underscore/hyphen (must start with a letter).
    Anything else raises ValueError — callers should map that to HTTP 400."""
    out: list[str] = []
    for k, v in (overrides or {}).items():
        if not isinstance(k, str) or not _OVERRIDE_KEY_RE.fullmatch(k):
            raise ValueError(f"invalid override key {k!r}: must match [a-z][a-z0-9_-]*")
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                out.append(flag)
        elif isinstance(v, (list, tuple)):
            for item in v:
                out += [flag, str(item)]
        elif v is not None:
            out += [flag, str(v)]
    return out


class Job:
    __slots__ = ("id", "stage", "project_dir", "dir", "command", "status", "pid",
                 "pid_start_time", "started_at", "ended_at", "exit_code", "_proc", "_console_fh")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def meta(self) -> dict:
        return {"job_id": self.id, "stage": self.stage, "project": str(self.project_dir),
                "dir": str(self.dir), "command": self.command, "status": self.status,
                "pid": self.pid, "pid_start_time": self.pid_start_time,
                "started_at": self.started_at, "ended_at": self.ended_at,
                "exit_code": self.exit_code}

    def write_meta(self) -> None:
        (self.dir / "meta.json").write_text(json.dumps(self.meta(), indent=2), encoding="utf-8")


class JobManager:
    """Owns the running jobs and the on-disk job registry. One instance per server process."""

    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        # (job_id, since) -> (file_size_when_cached, byte_offset_to_resume_from, last_seq_emitted)
        # Lets read_events skip rescanning the prefix on each poll (otherwise the SSE generator is
        # quadratic in the file size for a long-running synth/train).
        self._read_offsets: dict[tuple[str, int], tuple[int, int, int]] = {}

    # ---- discovery ----

    def scan(self, project_dirs) -> None:
        """(Re-)load job metadata from disk for the given projects. Marks dead 'running' jobs as 'crashed'."""
        for pd in project_dirs:
            jobs_root = Path(pd) / "dataset" / "jobs"
            if not jobs_root.is_dir():
                continue
            for jd in sorted(jobs_root.iterdir()):
                meta_f = jd / "meta.json"
                if not meta_f.is_file():
                    continue
                try:
                    m = json.loads(meta_f.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue
                jid = m.get("job_id") or jd.name
                with self._lock:
                    if jid in self._jobs and self._jobs[jid]._proc is not None:
                        continue  # live job we own — trust memory
                    j = Job(id=jid, stage=m.get("stage"), project_dir=Path(m.get("project", pd)),
                            dir=jd, command=m.get("command"), status=m.get("status"),
                            pid=m.get("pid"), pid_start_time=m.get("pid_start_time"),
                            started_at=m.get("started_at"),
                            ended_at=m.get("ended_at"), exit_code=m.get("exit_code"),
                            _proc=None, _console_fh=None)
                    if j.status == "running":
                        # PID alone is not enough — after a server restart, the OS may have
                        # recycled it to an unrelated process. If we recorded a start time,
                        # require it to match (within 2s; clock-tick rounding on some systems).
                        alive = _pid_alive(j.pid)
                        if alive and j.pid_start_time is not None:
                            now_start = _proc_start_time(j.pid)
                            if now_start is not None and abs(now_start - float(j.pid_start_time)) > 2.0:
                                alive = False  # different process now wearing this PID
                        if not alive:
                            j.status = "crashed"
                            j.ended_at = j.ended_at or _now_iso()
                            try:
                                j.write_meta()
                            except OSError:
                                pass
                    self._jobs[jid] = j

    def list(self, project_dir=None) -> list[dict]:
        with self._lock:
            jobs = list(self._jobs.values())
        if project_dir is not None:
            pd = str(Path(project_dir).resolve())
            jobs = [j for j in jobs if str(Path(j.project_dir).resolve()) == pd]
        jobs.sort(key=lambda j: j.started_at or "", reverse=True)
        return [j.meta() for j in jobs]

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            j = self._jobs.get(job_id)
        return j.meta() if j else None

    def running_for(self, project_dir, stage) -> str | None:
        pd = str(Path(project_dir).resolve())
        with self._lock:
            return self._running_for_locked(pd, stage)

    def _running_for_locked(self, project_dir_resolved: str, stage: str) -> str | None:
        for j in self._jobs.values():
            if j.status == "running" and j.stage == stage and str(Path(j.project_dir).resolve()) == project_dir_resolved:
                return j.id
        return None

    # ---- launch / control ----

    def start(self, project_dir, stage: str, *, overrides: dict | None = None,
              smoke: bool = False, gpu: int | None = None,
              extra_args: list[str] | None = None) -> dict:
        if stage not in _STAGES:
            raise ValueError(f"unknown stage {stage!r}")
        project_dir = Path(project_dir).expanduser().resolve()
        if not (project_dir / "project.toml").is_file():
            raise FileNotFoundError(f"no project.toml at {project_dir}")

        argv = [sys.executable, "-u", "-m", f"pipeline.{stage}", "--project", str(project_dir)]
        if gpu is not None and stage == "train":
            argv += ["--gpu", str(gpu)]
        if smoke and stage == "train":
            argv.append("--smoke")          # only `train` has a --smoke flag today
        argv += _flags_from_overrides(overrides)         # may raise ValueError on bad keys
        if extra_args:
            argv += list(extra_args)

        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        repo_root = Path(__file__).resolve().parent.parent  # the dir containing the `pipeline/` package
        pd_resolved = str(project_dir)

        # Claim-the-stage and spawn under the lock so two concurrent clients can't both win the
        # "is anything running for this (project, stage)?" check before either has assigned itself.
        with self._lock:
            if stage in _EXCLUSIVE:
                busy = self._running_for_locked(pd_resolved, stage)
                if busy:
                    raise RuntimeError(f"stage {stage!r} already running for this project (job {busy})")
            jid = _new_job_id()
            jdir = project_dir / "dataset" / "jobs" / jid
            jdir.mkdir(parents=True, exist_ok=True)
            events_path = jdir / "events.ndjson"
            console_path = jdir / "console.log"
            env["VOICEPIPE_EVENTS_FILE"] = str(events_path)
            console_fh = open(console_path, "w", buffering=1, encoding="utf-8")
            # start_new_session=True -> new process group, so we can signal the whole tree on cancel
            proc = subprocess.Popen(argv, cwd=str(repo_root),
                                     stdout=console_fh, stderr=subprocess.STDOUT, env=env,
                                     start_new_session=True)
            j = Job(id=jid, stage=stage, project_dir=project_dir, dir=jdir, command=argv,
                    status="running", pid=proc.pid,
                    pid_start_time=_proc_start_time(proc.pid),
                    started_at=_now_iso(), ended_at=None,
                    exit_code=None, _proc=proc, _console_fh=console_fh)
            j.write_meta()
            self._jobs[jid] = j
        threading.Thread(target=self._reap, args=(j,), daemon=True).start()
        return j.meta()

    def _reap(self, j: Job) -> None:
        code = j._proc.wait()
        try:
            j._console_fh.close()
        except Exception:
            pass
        j.exit_code = code
        j.ended_at = _now_iso()
        # If the stage process died without emitting a stage_end (e.g. killed, or crashed before
        # its try-block), append one so consumers always see a terminal event.
        if not _last_event_is_terminal(j.dir / "events.ndjson"):
            try:
                with (j.dir / "events.ndjson").open("a", encoding="utf-8") as f:
                    status = "cancelled" if code in (-signal.SIGTERM, -signal.SIGKILL, 143, 137) else ("ok" if code == 0 else "error")
                    f.write(json.dumps({"ts": _now_iso(), "stage": j.stage, "type": "stage_end",
                                        "status": status, "exit_code": code,
                                        "error": None if status != "error" else f"process exited {code} without a stage_end"}) + "\n")
            except OSError:
                pass
        j.status = ("cancelled" if code in (-signal.SIGTERM, -signal.SIGKILL, 143, 137)
                    else "ok" if code == 0 else "error")
        try:
            j.write_meta()
        except OSError:
            pass

    def cancel(self, job_id: str, grace: float = 5.0) -> dict:
        with self._lock:
            j = self._jobs.get(job_id)
        if not j:
            raise KeyError(job_id)
        if j.status != "running" or j._proc is None:
            return j.meta()
        try:
            os.killpg(os.getpgid(j._proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        def _hard_kill():
            time.sleep(grace)
            if j._proc.poll() is None:
                try:
                    os.killpg(os.getpgid(j._proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        threading.Thread(target=_hard_kill, daemon=True).start()
        return j.meta()

    def kill_all(self, *, grace: float = 2.0) -> int:
        """SIGTERM every running job's process group, briefly wait, then SIGKILL stragglers.
        Returns the number of jobs we signalled. Called from the server's shutdown handler so an
        orphaned child stage isn't left running when the supervising server exits."""
        with self._lock:
            running = [j for j in self._jobs.values()
                       if j.status == "running" and j._proc is not None]
        if not running:
            return 0
        for j in running:
            try:
                os.killpg(os.getpgid(j._proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                pass
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline and any(j._proc.poll() is None for j in running):
            time.sleep(0.1)
        for j in running:
            if j._proc.poll() is None:
                try:
                    os.killpg(os.getpgid(j._proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
        return len(running)

    # ---- event streaming ----

    def read_events(self, job_id: str, since: int = 0) -> list[dict]:
        """All events with seq > `since`. seq is the 1-based line number in events.ndjson.
        Caches a byte offset per (job_id, since) so repeated polls (the SSE loop) don't re-scan
        the file prefix each time — that would be O(file_size) per poll."""
        with self._lock:
            j = self._jobs.get(job_id)
        if not j:
            raise KeyError(job_id)
        path = j.dir / "events.ndjson"
        cache_key = (job_id, since)
        with self._lock:
            cached = self._read_offsets.get(cache_key)
        try:
            size = path.stat().st_size if path.is_file() else 0
        except OSError:
            size = 0
        start_offset, start_seq = 0, since
        if cached and cached[0] <= size:
            _, start_offset, start_seq = cached
        new_events, new_offset, new_seq = _read_events_from_offset(path, start_offset, start_seq, since)
        with self._lock:
            self._read_offsets[cache_key] = (size, new_offset, new_seq)
        return new_events

    def is_running(self, job_id: str) -> bool:
        with self._lock:
            j = self._jobs.get(job_id)
        return bool(j and j.status == "running")


# ---- module-level helpers (no manager state) ----

def _read_events_file(path: Path, since: int = 0) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    try:
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if i <= since or not line.strip():
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            evt["seq"] = i
            out.append(evt)
    except OSError:
        pass
    return out


def _read_events_from_offset(path: Path, start_offset: int, start_seq: int,
                             since: int) -> tuple[list[dict], int, int]:
    """Resume reading at byte `start_offset` (a line boundary), continuing seq numbering from
    `start_seq`. Returns (new_events, next_offset, last_seq). On any unexpected condition (file
    shrank, mid-line offset, decode failure) falls back to a full scan from the start."""
    if not path.is_file():
        return [], 0, start_seq
    out: list[dict] = []
    seq = start_seq
    try:
        with path.open("rb") as f:
            try:
                f.seek(start_offset)
            except OSError:
                f.seek(0); seq = 0
            buf = f.read()
            pos = f.tell()
        text = buf.decode("utf-8", errors="replace")
        consumed_lines = text.split("\n")
        # The last element is the partial trailing line (or empty if file ended in \n);
        # we only consume complete lines and advance the offset accordingly.
        complete = consumed_lines[:-1]
        trailing = consumed_lines[-1]
        next_offset = pos - len(trailing.encode("utf-8"))
        for line in complete:
            seq += 1
            s = line.strip()
            if seq <= since or not s:
                continue
            try:
                evt = json.loads(s)
            except json.JSONDecodeError:
                continue
            evt["seq"] = seq
            out.append(evt)
        return out, next_offset, seq
    except OSError:
        # I/O failed mid-read; fall back to a full scan so the caller still makes progress.
        full = _read_events_file(path, since)
        last = full[-1]["seq"] if full else since
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        return full, size, last


def _last_event_is_terminal(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except OSError:
        return False
    if not lines:
        return False
    try:
        return json.loads(lines[-1]).get("type") == "stage_end"
    except json.JSONDecodeError:
        return False


def read_console(path: Path, tail: int | None = None) -> str:
    if not Path(path).is_file():
        return ""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if tail:
        return "\n".join(text.splitlines()[-tail:])
    return text
