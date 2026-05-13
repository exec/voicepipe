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

import json
import os
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


def _flags_from_overrides(overrides: dict | None) -> list[str]:
    """{"max_seq_len": 4096, "no_downsample": true, "model": "x"} -> ["--max-seq-len","4096","--no-downsample","--model","x"]"""
    out: list[str] = []
    for k, v in (overrides or {}).items():
        flag = "--" + str(k).replace("_", "-")
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
                 "started_at", "ended_at", "exit_code", "_proc", "_console_fh")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def meta(self) -> dict:
        return {"job_id": self.id, "stage": self.stage, "project": str(self.project_dir),
                "dir": str(self.dir), "command": self.command, "status": self.status,
                "pid": self.pid, "started_at": self.started_at, "ended_at": self.ended_at,
                "exit_code": self.exit_code}

    def write_meta(self) -> None:
        (self.dir / "meta.json").write_text(json.dumps(self.meta(), indent=2), encoding="utf-8")


class JobManager:
    """Owns the running jobs and the on-disk job registry. One instance per server process."""

    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

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
                            pid=m.get("pid"), started_at=m.get("started_at"),
                            ended_at=m.get("ended_at"), exit_code=m.get("exit_code"),
                            _proc=None, _console_fh=None)
                    if j.status == "running" and not _pid_alive(j.pid):
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
            for j in self._jobs.values():
                if j.status == "running" and j.stage == stage and str(Path(j.project_dir).resolve()) == pd:
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
        if stage in _EXCLUSIVE:
            busy = self.running_for(project_dir, stage)
            if busy:
                raise RuntimeError(f"stage {stage!r} already running for this project (job {busy})")

        jid = _new_job_id()
        jdir = project_dir / "dataset" / "jobs" / jid
        jdir.mkdir(parents=True, exist_ok=True)
        events_path = jdir / "events.ndjson"
        console_path = jdir / "console.log"

        argv = [sys.executable, "-u", "-m", f"pipeline.{stage}", "--project", str(project_dir)]
        if gpu is not None and stage == "train":
            argv += ["--gpu", str(gpu)]
        if smoke and stage == "train":
            argv.append("--smoke")          # only `train` has a --smoke flag today
        argv += _flags_from_overrides(overrides)
        if extra_args:
            argv += list(extra_args)

        env = dict(os.environ)
        env["VOICEPIPE_EVENTS_FILE"] = str(events_path)
        env["PYTHONUNBUFFERED"] = "1"

        console_fh = open(console_path, "w", buffering=1, encoding="utf-8")
        repo_root = Path(__file__).resolve().parent.parent  # the dir containing the `pipeline/` package
        # start_new_session=True -> new process group, so we can signal the whole tree on cancel
        proc = subprocess.Popen(argv, cwd=str(repo_root),
                                 stdout=console_fh, stderr=subprocess.STDOUT, env=env,
                                 start_new_session=True)
        j = Job(id=jid, stage=stage, project_dir=project_dir, dir=jdir, command=argv,
                status="running", pid=proc.pid, started_at=_now_iso(), ended_at=None,
                exit_code=None, _proc=proc, _console_fh=console_fh)
        j.write_meta()
        with self._lock:
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

    # ---- event streaming ----

    def read_events(self, job_id: str, since: int = 0) -> list[dict]:
        """All events with seq > `since`. seq is the 1-based line number in events.ndjson."""
        with self._lock:
            j = self._jobs.get(job_id)
        if not j:
            raise KeyError(job_id)
        return _read_events_file(j.dir / "events.ndjson", since)

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
