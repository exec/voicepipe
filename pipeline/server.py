"""
voicepipe control server — a small local HTTP API + the web GUI, in front of the (unchanged)
pipeline engine.

Run it:
    voicepipe serve                       # http://127.0.0.1:8765 , no auth (loopback)
    voicepipe serve --host 0.0.0.0 --auth-token SECRET    # remote access, password-gated
    voicepipe serve --unix-socket /run/voicepipe.sock     # no TCP port at all (auth bypassed)

Design notes:
  - The engine stays 100% the CLI. A stage run is a subprocess (`pipeline.jobs`); this process
    never imports `pipeline.train` etc.
  - Auth: a single shared token. Required on /v1/* when `--auth-token`/$VOICEPIPE_AUTH_TOKEN is
    set OR when bound to a non-loopback host. Bypassed entirely on a Unix socket (the native
    desktop app uses that path and is trusted by construction). The static UI is always served
    so the login prompt can load.
  - Projects are discovered by scanning "roots" (one level deep for a project.toml) plus an
    explicit registry file (~/.config/voicepipe/registry.json) for dirs outside the roots.
"""

import argparse
import asyncio
import json
import os
import re
import secrets
import sys
import time
from dataclasses import asdict
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException, Request, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, ConfigDict
except ImportError as e:  # pragma: no cover
    raise SystemExit("the GUI server needs FastAPI + uvicorn — `pip install -e .[gui]`") from e

from pipeline import jobs as jobsmod
from pipeline import scaffold
from pipeline import util as utilmod
from pipeline.project import load_project, Project

_REPO_ROOT = Path(__file__).resolve().parent.parent
_WEBUI_DIR = Path(__file__).resolve().parent / "webui"
_CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "voicepipe"
_REGISTRY_FILE = _CONFIG_DIR / "registry.json"
_CONFIG_FILE = _CONFIG_DIR / "config.json"      # { "project_roots": [ ... ] }
_ENV_FILE = _CONFIG_DIR / "env"


def _load_app_config() -> dict:
    if _CONFIG_FILE.is_file():
        try:
            return json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_app_config(cfg: dict) -> None:
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def _default_roots() -> list[Path]:
    """Where to look for projects when neither --root nor config.json says otherwise:
    the launch directory (skipped if it's / or $HOME — too broad), plus ~/voicepipe-projects.
    That's it. Anything else lives in the registry (created/opened projects) or config.json."""
    out: list[Path] = []
    cwd = Path.cwd().resolve()
    if cwd not in (Path("/"), Path.home().resolve()):
        out.append(cwd)
    out.append(Path.home() / "voicepipe-projects")
    return out


def _seed_registry_from_repo() -> None:
    """If we're running from a source checkout (the `pipeline` package's parent has projects/ or
    scratch/ with project.toml dirs), register those once so they don't vanish from the list when
    we stopped baking them into the defaults. No-op for a packaged install (no such dirs)."""
    for sub in ("projects", "scratch"):
        base = _REPO_ROOT / sub
        if not base.is_dir():
            continue
        for child in base.iterdir():
            if child.is_dir() and (child / "project.toml").is_file():
                _register_path(child)


def _resolve_roots(cli_roots: list | None) -> list[Path]:
    """--root args win outright; otherwise config.json's project_roots, else the built-in defaults.
    Plus dirs explicitly registered via the registry are always scanned (handled by Registry)."""
    if cli_roots:
        return [Path(r).expanduser().resolve() for r in cli_roots]
    cfg_roots = _load_app_config().get("project_roots")
    if cfg_roots:
        return [Path(r).expanduser().resolve() for r in cfg_roots]
    return [r.expanduser().resolve() for r in _default_roots()]


def _start_parent_watchdog() -> None:
    """If $VOICEPIPE_PARENT_PID is set (the desktop shell sets it), exit when that parent dies —
    so a force-killed app never leaves the engine orphaned. No-op for a plain `voicepipe serve`."""
    pp = os.environ.get("VOICEPIPE_PARENT_PID")
    if not pp or not pp.isdigit():
        return
    parent = int(pp)
    import threading, time as _t
    def _watch():
        while True:
            _t.sleep(2)
            if os.getppid() != parent:        # parent gone -> we got reparented (to launchd/init)
                os._exit(0)
    threading.Thread(target=_watch, daemon=True).start()


# The canonical env-file loader/writer pair lives in pipeline.util — keep thin wrappers here so
# call sites in this module read naturally. _ENV_FILE (above) is kept as a UX label only.
def _load_env_file() -> None:
    utilmod.load_env_file()


def _set_env_file_var(key: str, value: str) -> None:
    """Persist `KEY=value` to ~/.config/voicepipe/env. Raises RuntimeError if the file can't be
    chmod'd to 0600 — secrets aren't allowed to live world-readable."""
    utilmod.set_env_file_var(key, value)

# Fields whose (long) string value is spilled to prompts/<field>.md and referenced by filename
# when the GUI writes a config — mirrors how a human authors project.toml.
_SPILL_FIELDS = {"synth_preamble", "variety_menus", "content_rules"}
_SPILL_THRESHOLD = 400  # chars; also spill anything containing a newline


# --------------------------------------------------------------------------- registry / discovery

def _load_registry() -> dict:
    if _REGISTRY_FILE.is_file():
        try:
            return json.loads(_REGISTRY_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"paths": []}


def _save_registry(reg: dict) -> None:
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _REGISTRY_FILE.write_text(json.dumps(reg, indent=2), encoding="utf-8")


def _register_path(path: Path) -> None:
    reg = _load_registry()
    p = str(Path(path).resolve())
    if p not in reg.get("paths", []):
        reg.setdefault("paths", []).append(p)
        _save_registry(reg)


def _unregister_path(path: Path) -> None:
    reg = _load_registry()
    p = str(Path(path).resolve())
    if p in reg.get("paths", []):
        reg["paths"] = [x for x in reg["paths"] if x != p]
        _save_registry(reg)


def _unhide_project(path: Path) -> None:
    """Reverse a DELETE /v1/projects (un-mask a project that was hidden from the list)."""
    cfg = _load_app_config()
    hidden = cfg.get("hidden_projects")
    if not hidden:
        return
    p = str(Path(path).resolve())
    if p in hidden:
        cfg["hidden_projects"] = [x for x in hidden if x != p]
        _save_app_config(cfg)


def _slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-") or "project"


class Registry:
    """Maps slug -> project dir. Rebuilt from the configured roots + the on-disk registry."""

    def __init__(self, roots: list[Path]):
        self.roots = [Path(r).expanduser().resolve() for r in roots]
        self._slug_to_dir: dict[str, Path] = {}

    def _discover_dirs(self) -> list[Path]:
        seen: list[Path] = []
        for root in self.roots:
            if not root.is_dir():
                continue
            if (root / "project.toml").is_file():
                seen.append(root)
            for child in sorted(root.iterdir()):
                if child.is_dir() and (child / "project.toml").is_file():
                    seen.append(child)
        for p in _load_registry().get("paths", []):
            pp = Path(p)
            if (pp / "project.toml").is_file():
                seen.append(pp.resolve())
        hidden = {Path(h).resolve() for h in _load_app_config().get("hidden_projects", [])}
        # de-dupe preserving order, dropping anything explicitly hidden
        out, known = [], set()
        for p in seen:
            rp = p.resolve()
            if rp not in known and rp not in hidden:
                known.add(rp)
                out.append(rp)
        return out

    def refresh(self) -> list[Path]:
        dirs = self._discover_dirs()
        self._slug_to_dir = {}
        for d in dirs:
            base = _slugify(d.name)
            slug = base
            i = 2
            while slug in self._slug_to_dir:
                slug = f"{base}-{i}"
                i += 1
            self._slug_to_dir[slug] = d
        return dirs

    def dir_for(self, slug: str) -> Path:
        if slug not in self._slug_to_dir:
            self.refresh()
        if slug not in self._slug_to_dir:
            raise KeyError(slug)
        return self._slug_to_dir[slug]

    def slug_for(self, d: Path) -> str:
        d = Path(d).resolve()
        for slug, pd in self._slug_to_dir.items():
            if pd.resolve() == d:
                return slug
        self.refresh()
        for slug, pd in self._slug_to_dir.items():
            if pd.resolve() == d:
                return slug
        return _slugify(d.name)

    def all(self) -> list[tuple[str, Path]]:
        self.refresh()
        return sorted(self._slug_to_dir.items())


# --------------------------------------------------------------------------- project serialization

def _jsonable(v):
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def _project_config_dict(proj: Project) -> dict:
    d = asdict(proj)
    d.pop("root", None)
    return _jsonable(d)


def _dataset_state(proj: Project) -> dict:
    def count_lines(p: Path) -> int:
        try:
            return sum(1 for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip())
        except OSError:
            return 0
    dd = proj.dataset_path
    raw_dir = dd("raw")
    raw_batches = sorted(raw_dir.glob("batch_*.jsonl")) if raw_dir.is_dir() else []
    state = {
        "raw": {"batches": len(raw_batches),
                "pairs": sum(count_lines(p) for p in raw_batches)},
        "dedup": {"pairs": count_lines(dd("dedup", "pairs.jsonl"))},
        "triage": {"scored": count_lines(dd("triage", "scored.jsonl")),
                   "kept": count_lines(dd("triage", "keep.jsonl"))},
        "final": {"train": count_lines(dd("final", "train.jsonl")),
                  "val": count_lines(dd("final", "val.jsonl"))},
        "adapter": {"has_adapter": (dd("adapter", "final").is_dir()),
                    "has_gguf": bool(list(dd("adapter").glob("*.gguf"))) if dd("adapter").is_dir() else False},
    }
    return state


def _project_summary(reg: Registry, d: Path) -> dict:
    try:
        proj = load_project(d)
        name, desc = proj.name, proj.description
    except Exception as e:  # noqa: BLE001 - report broken projects rather than failing the list
        return {"id": reg.slug_for(d), "path": str(d), "name": d.name, "description": "",
                "error": f"{type(e).__name__}: {e}"}
    return {"id": reg.slug_for(d), "path": str(d), "name": name, "description": desc}


def _project_detail(reg: Registry, d: Path) -> dict:
    proj = load_project(d)
    corpus = proj.corpus_path()
    corpus_files = sorted(p.name for p in corpus.iterdir()
                          if p.is_file() and p.suffix.lower() in (".txt", ".md")) if corpus.is_dir() else []
    seeds_n = 0
    if proj.seeds_file and proj.p(proj.seeds_file).is_file():
        seeds_n = sum(1 for ln in proj.p(proj.seeds_file).read_text(encoding="utf-8").splitlines() if ln.strip())
    return {
        "id": reg.slug_for(d), "path": str(d), "name": proj.name, "description": proj.description,
        "config": _project_config_dict(proj),
        "corpus_files": corpus_files, "seed_count": seeds_n,
        "dataset_state": _dataset_state(proj),
    }


def _write_config(d: Path, incoming: dict) -> None:
    """Merge `incoming` onto the project's current config and write project.toml back, atomically:
    the new file is written to a temp path and validated (via load_project) before it replaces the
    original, so a bad config never corrupts the project. Long prose fields are spilled to
    prompts/<field>.md and referenced by filename. (TOML has no null — None values are dropped and
    fall back to dataclass defaults on load.)"""
    _dump = _toml_dumps
    toml_path = d / "project.toml"
    if sys.version_info >= (3, 11):
        import tomllib
    else:  # pragma: no cover
        import tomli as tomllib
    try:
        current = tomllib.loads(toml_path.read_text(encoding="utf-8")) if toml_path.is_file() else {}
    except tomllib.TOMLDecodeError:
        current = {}  # existing file is broken — rebuild from the incoming config alone

    def _spill(text: str, fname: str) -> str:
        target = d / fname
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        return fname

    def _is_prose(v) -> bool:
        return isinstance(v, str) and (len(v) > _SPILL_THRESHOLD or "\n" in v)

    for k, v in (incoming or {}).items():
        if k == "root":
            continue
        if k in _SPILL_FIELDS and _is_prose(v):
            current[k] = _spill(v, f"prompts/{k}.md")
        elif k == "modes" and isinstance(v, list):
            out_modes = []
            for m in v:
                m = dict(m)
                if _is_prose(m.get("description", "")):
                    safe = _slugify(m.get("name", "mode"))
                    m["description"] = _spill(m["description"], f"prompts/modes/{safe}.md")
                out_modes.append(m)
            current["modes"] = out_modes
        elif k == "triage" and isinstance(v, dict):
            t = dict(v)
            if _is_prose(t.get("rubric", "")):
                t["rubric"] = _spill(t["rubric"], "prompts/triage_rubric.md")
            current["triage"] = {**current.get("triage", {}), **t}
        elif k == "deploy" and isinstance(v, dict):
            dp = dict(v)
            if _is_prose(dp.get("system_message", "")):
                dp["system_message"] = _spill(dp["system_message"], "prompts/deploy_system.md")
            current["deploy"] = {**current.get("deploy", {}), **dp}
        elif isinstance(v, dict) and isinstance(current.get(k), dict):
            current[k] = {**current[k], **v}     # shallow-merge stage configs
        else:
            current[k] = v

    # write to a temp file, validate it loads, then atomically replace the original
    tmp = toml_path.with_suffix(".toml.tmp")
    tmp.write_text(_dump(current), encoding="utf-8")
    try:
        load_project(tmp)              # validate the new content (passing the file path directly)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(toml_path)


def _toml_dumps(obj, _prefix="") -> str:
    """Minimal TOML serializer (no external dep). Handles scalars, strings (incl. multiline via
    triple-quote), lists of scalars, arrays of inline tables, and nested tables / array-of-tables.
    Lossy: comments and literal-string quoting in an existing project.toml are not preserved."""
    def fmt_scalar(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return repr(v)
        if isinstance(v, str):
            if "\n" in v:
                return '"""\n' + v + ('"""' if v.endswith("\n") else '\n"""')
            return json.dumps(v)
        return json.dumps(v)

    def fmt_inline_table(t):
        return "{ " + ", ".join(f"{k} = {fmt_scalar(x)}" for k, x in t.items() if x is not None) + " }"

    def _clean(d):  # TOML has no null — drop None keys (they fall back to dataclass defaults on load)
        return {k: v for k, v in d.items() if v is not None}

    lines, tables, arrays_of_tables = [], [], []
    for k, v in obj.items():
        if v is None:
            continue
        if isinstance(v, dict):
            tables.append((k, _clean(v)))
        elif isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            cleaned = [_clean(x) for x in v]
            if all(all(not isinstance(xx, (dict, list)) for xx in x.values()) for x in cleaned):
                lines.append(f"{k} = [\n" + "".join(f"  {fmt_inline_table(x)},\n" for x in cleaned) + "]")
            else:
                arrays_of_tables.append((k, cleaned))
        elif isinstance(v, list):
            lines.append(f"{k} = [" + ", ".join(fmt_scalar(x) for x in v if x is not None) + "]")
        else:
            lines.append(f"{k} = {fmt_scalar(v)}")
    out = "\n".join(lines)
    for k, t in tables:
        out += f"\n\n[{(_prefix + '.' if _prefix else '') + k}]\n" + _toml_dumps(t, _prefix=(_prefix + '.' if _prefix else '') + k)
    for k, lst in arrays_of_tables:
        for item in lst:
            out += f"\n\n[[{(_prefix + '.' if _prefix else '') + k}]]\n" + _toml_dumps(item, _prefix=(_prefix + '.' if _prefix else '') + k)
    return out.strip() + "\n"


# --------------------------------------------------------------------------- request schemas

class _ConfigPut(BaseModel):
    """PUT /v1/config — the known keys. Unknown keys are rejected (422)."""
    model_config = ConfigDict(extra="forbid")
    project_roots: list[str] | None = None
    llama_cpp_dir: str | None = None
    ollama_api_key: str | None = None


class _NewProject(BaseModel):
    """POST /v1/projects — either register an existing dir (path) or scaffold a new one
    (name+template). Unknown keys are rejected so a typo doesn't silently degrade to a scaffold."""
    model_config = ConfigDict(extra="forbid")
    path: str | None = None
    name: str | None = None
    description: str | None = ""
    template: str | None = "character"
    parent_dir: str | None = None


class _FilePut(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content: str = ""


class _RunStage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    smoke: bool | None = False
    gpu: int | None = None
    overrides: dict[str, object] | None = None


# Project-config PUT body is the full Project dataclass (huge surface), so it stays a free-form
# dict — but we validate it round-trips through load_project before accepting it (existing
# behaviour, just kept explicit here).
class _ProjectConfigPut(BaseModel):
    model_config = ConfigDict(extra="allow")
    config: dict | None = None


# What we'll accept as a "project root" via POST /v1/projects { path }. Absolute paths must live
# under one of these, or be a configured project root, or already be on the registry. Symlinks-
# to-elsewhere are caught by resolving with strict semantics before the check.
def _allowed_project_roots(reg: "Registry") -> list[Path]:
    """The dirs an absolute `path` is allowed to live under for POST /v1/projects."""
    roots: list[Path] = []
    roots.extend(reg.roots)
    roots.append(Path.home().resolve())
    return [r.resolve() for r in roots]


def _path_within_any(p: Path, roots: list[Path]) -> bool:
    p = p.resolve()
    for r in roots:
        try:
            p.relative_to(r)
            return True
        except ValueError:
            continue
    return False


# Allow-list for what put_file can write. Editing project source (prompts, corpus, seeds,
# project.toml itself) is in scope; writing arbitrary binaries is not.
_PUT_FILE_EXTS = {".md", ".txt", ".jsonl", ".toml"}
_PUT_FILE_MAX_BYTES = 1 * 1024 * 1024  # 1 MiB


# --------------------------------------------------------------------------- the app

def create_app(roots: list[Path], *, auth_token: str | None = None, auth_required: bool = False,
               bind_host: str | None = None) -> FastAPI:
    app = FastAPI(title="voicepipe", version="0.1.0")
    # CORS: always allow loopback origins, regardless of bind. The web UI is same-origin (served
    # by this process) when accessed locally, AND a remote worker's legitimate cross-origin caller
    # is *also* a loopback origin — the user's local GUI calling out to a public worker. Locking
    # this down to same-origin-only when bound non-loopback breaks the remote-engine flow without
    # any security benefit (the bearer token, not the origin, is the auth gate). No wildcards, no
    # credentials cross-site — just loopback origins, always.
    app.add_middleware(CORSMiddleware,
                       allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:[0-9]+)?$",
                       allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                       allow_headers=["Authorization", "Content-Type", "X-Auth-Token"])
    reg = Registry(roots)
    mgr = jobsmod.JobManager()
    mgr.scan([d for _, d in reg.all()])

    # Cached job scan — the SSE polling loop + the GUI's 4s job-list refresh shouldn't serialize
    # on a full disk walk under the JobManager lock. Invalidated on start/cancel + after _SCAN_TTL.
    _SCAN_TTL = 10.0
    _scan_state = {"ts": 0.0}

    def _scan_jobs(force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - _scan_state["ts"]) < _SCAN_TTL:
            return
        mgr.scan([d for _, d in reg.all()])
        _scan_state["ts"] = now

    need_auth = bool(auth_token) and auth_required

    # Short-lived SSE tickets: EventSource can't set Authorization headers, so an authenticated
    # client GETs /v1/sse-ticket and appends ?ticket=... to the events URL. The ticket is single-
    # use and expires in 30s. This is the documented exception to "no tokens in URLs".
    _tickets: dict[str, float] = {}
    _TICKET_TTL = 30.0

    def _make_ticket() -> str:
        t = secrets.token_urlsafe(24)
        _tickets[t] = time.monotonic() + _TICKET_TTL
        # opportunistic GC of expired tickets
        now = time.monotonic()
        for k in [k for k, exp in _tickets.items() if exp < now]:
            _tickets.pop(k, None)
        return t

    def _consume_ticket(t: str) -> bool:
        exp = _tickets.pop(t, None)
        return bool(exp) and exp > time.monotonic()

    # /v1/health is always reachable (liveness probe; lets the UI learn auth_required before anything else).
    _AUTH_EXEMPT = ("/v1/health",)

    @app.middleware("http")
    async def _auth(request: Request, call_next):
        path = request.url.path
        if need_auth and path.startswith("/v1") and path not in _AUTH_EXEMPT and request.method != "OPTIONS":
            # SSE events endpoint also accepts a single-use ticket on the query string (because
            # EventSource can't set headers). Every other endpoint must use Authorization.
            if path.startswith("/v1/jobs/") and path.endswith("/events"):
                tk = request.query_params.get("ticket") or ""
                if tk and _consume_ticket(tk):
                    return await call_next(request)
            tok = ""
            authz = request.headers.get("authorization", "")
            if authz.lower().startswith("bearer "):
                tok = authz[7:].strip()
            if not tok:
                tok = request.headers.get("x-auth-token", "") or request.cookies.get("vp_token", "")
            # constant-time compare; both sides are str
            if not secrets.compare_digest(tok or "", auth_token or ""):
                return JSONResponse({"detail": "unauthorized"}, status_code=401)
        return await call_next(request)

    @app.get("/v1/sse-ticket")
    def sse_ticket():
        """Mint a single-use, 30s-TTL token for the SSE events endpoint (EventSource can't set
        an Authorization header). The ticket is consumed on first use. No-op-ish but harmless
        when auth isn't required."""
        return {"ticket": _make_ticket(), "ttl_sec": _TICKET_TTL}

    # ---- meta ----

    @app.get("/v1/health")
    def health():
        try:
            import torch  # noqa: F401
            has_torch = True
        except Exception:
            has_torch = False
        import shutil as _sh
        return {"ok": True, "version": "0.1.0", "python": sys.version.split()[0],
                "has_torch": has_torch, "has_ollama": bool(_sh.which("ollama")),
                "ollama_api_key_set": bool(os.environ.get("OLLAMA_API_KEY")),
                "auth_required": need_auth, "roots": [str(r) for r in reg.roots]}

    @app.get("/v1/defaults")
    def defaults():
        return _project_config_dict(_jsonable_default_project())

    @app.get("/v1/templates")
    def templates():
        return scaffold.list_templates()

    @app.get("/v1/providers")
    def providers():
        # Minimal for now: report the Ollama Cloud entry and whether its key is present.
        return [{"id": "ollama_cloud", "kind": "ollama_cloud", "base_url": "https://ollama.com",
                 "ready": bool(os.environ.get("OLLAMA_API_KEY")),
                 "note": "set OLLAMA_API_KEY to use synthesis/triage"},
                {"id": "ollama_local", "kind": "openai_compat", "base_url": "http://localhost:11434/v1",
                 "ready": True, "note": "local Ollama (also serves nomic-embed-text for dedup)"}]

    def _settings_payload() -> dict:
        return {"project_roots": [str(r) for r in reg.roots],
                "roots_are_default": not _load_app_config().get("project_roots"),
                "ollama_api_key_set": bool(os.environ.get("OLLAMA_API_KEY")),
                "llama_cpp_dir": _load_app_config().get("llama_cpp_dir") or "",
                "config_file": str(_CONFIG_FILE), "env_file": str(_ENV_FILE)}

    @app.get("/v1/config")
    def get_config():
        """All user-level settings: project-scan folders, whether the Ollama key is set, the
        default llama.cpp dir (for `deploy`), where the config/env files live."""
        return _settings_payload()

    @app.put("/v1/config")
    def put_config(body: _ConfigPut):
        cfg = _load_app_config()
        provided = body.model_dump(exclude_unset=True)
        if "project_roots" in provided:
            roots_in = provided["project_roots"] or []
            new_roots = [Path(str(r)).expanduser().resolve() for r in roots_in if str(r).strip()]
            cfg["project_roots"] = [str(r) for r in new_roots]
            reg.roots = new_roots
            reg.refresh()
            _scan_jobs(force=True)
        if "llama_cpp_dir" in provided:
            v = str(provided["llama_cpp_dir"] or "").strip()
            if v:
                cfg["llama_cpp_dir"] = str(Path(v).expanduser())
            else:
                cfg.pop("llama_cpp_dir", None)
        _save_app_config(cfg)
        if "ollama_api_key" in provided:                   # write to ~/.config/voicepipe/env (mode 600)
            try:
                _set_env_file_var("OLLAMA_API_KEY", str(provided["ollama_api_key"] or "").strip())
            except RuntimeError as e:
                raise HTTPException(500, f"couldn't persist API key securely: {e}")
        return _settings_payload()

    # ---- projects ----

    @app.get("/v1/projects")
    def list_projects():
        _scan_jobs()
        return [_project_summary(reg, d) for _, d in reg.all()]

    @app.post("/v1/projects")
    def create_project(body: _NewProject):
        if body.path:
            # Refuse paths that don't resolve to somewhere we expect projects to live: a
            # configured root, the user's home tree, or somewhere already on the registry.
            p_in = Path(body.path).expanduser()
            try:
                p = p_in.resolve(strict=True)
            except (OSError, RuntimeError):
                raise HTTPException(400, f"no such path: {body.path}")
            if not p.is_dir():
                raise HTTPException(400, f"not a directory: {p}")
            allowed = _allowed_project_roots(reg)
            already_registered = str(p) in [str(Path(x).resolve()) for x in _load_registry().get("paths", [])]
            if not (already_registered or _path_within_any(p, allowed)):
                raise HTTPException(400,
                    f"path {p} is not under a configured project root or your home directory; "
                    "add it as a project root in Settings first")
            if not (p / "project.toml").is_file():
                raise HTTPException(400, f"no project.toml at {p}")
            _register_path(p); _unhide_project(p)
            reg.refresh()
            return _project_detail(reg, p)
        # scaffold
        name = body.name
        if not name:
            raise HTTPException(400, "name (or path) required")
        template = body.template or "character"
        parent = Path(body.parent_dir or (Path.home() / "voicepipe-projects")).expanduser().resolve()
        dest = parent / _slugify(name)
        try:
            scaffold.create_project(dest, template=template, name=name, description=body.description or "")
        except (FileExistsError, ValueError) as e:
            raise HTTPException(400, str(e))
        _register_path(dest); _unhide_project(dest)
        reg.refresh()
        _scan_jobs(force=True)
        return _project_detail(reg, dest)

    def _dir(project_id: str) -> Path:
        try:
            return reg.dir_for(project_id)
        except KeyError:
            raise HTTPException(404, f"no project {project_id!r}")

    @app.get("/v1/projects/{project_id}")
    def get_project(project_id: str):
        return _project_detail(reg, _dir(project_id))

    @app.delete("/v1/projects/{project_id}")
    def remove_project(project_id: str, purge: bool = False):
        """Take a project off the list (unregister it + don't auto-rescan it). Files are left
        alone; with ?purge=true the produced dataset/ is deleted, but corpus/seeds/prompts stay."""
        d = _dir(project_id)
        _unregister_path(d)
        # also mask it: scan roots and the dev-checkout seeding would otherwise re-add it.
        cfg = _load_app_config()
        hidden = set(cfg.get("hidden_projects", []))
        hidden.add(str(d))
        cfg["hidden_projects"] = sorted(hidden)
        _save_app_config(cfg)
        if purge:
            import shutil
            ds = d / "dataset"
            if ds.is_dir():
                shutil.rmtree(ds, ignore_errors=True)
        reg.refresh()
        return {"ok": True, "removed": str(d), "purged": bool(purge)}

    @app.put("/v1/projects/{project_id}/config")
    async def put_project_config(project_id: str, request: Request):
        # The project config is the full Project dataclass — too sprawling for a flat Pydantic
        # schema. We do the validation that matters: it must be a JSON object, and the merged
        # result must load via load_project (already done inside _write_config). That keeps the
        # surface "any TOML-compatible dict" without letting truly broken shapes corrupt the file.
        d = _dir(project_id)
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(400, "body must be JSON")
        if not isinstance(body, dict):
            raise HTTPException(400, "body must be a JSON object")
        incoming = body.get("config", body) if isinstance(body.get("config", None), dict) else body
        if not isinstance(incoming, dict):
            raise HTTPException(400, "config must be a JSON object")
        try:
            _write_config(d, incoming)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, f"config did not validate: {type(e).__name__}: {e}")
        reg.refresh()
        return _project_detail(reg, d)

    def _safe_child(d: Path, relpath: str) -> Path:
        """Resolve `relpath` under project dir `d`, refusing anything that escapes it (incl. via
        `..` or a symlink). Also block the produced dataset/ subtree so file edits can't fight the
        engine."""
        root = d.resolve()
        fp = (root / relpath).resolve()
        try:
            fp.relative_to(root)
        except ValueError:
            raise HTTPException(400, "path escapes the project directory")
        try:
            from pipeline.project import load_project as _lp
            ds = _lp(d).dataset_path().resolve()
            if fp == ds or ds in fp.parents:
                raise HTTPException(400, "the dataset/ tree is managed by the engine; not editable here")
        except HTTPException:
            raise
        except Exception:
            pass
        return fp

    @app.get("/v1/projects/{project_id}/files/{relpath:path}")
    def get_file(project_id: str, relpath: str):
        fp = _safe_child(_dir(project_id), relpath)
        if not fp.is_file():
            raise HTTPException(404, "no such file")
        return PlainTextResponse(fp.read_text(encoding="utf-8", errors="replace"))

    @app.put("/v1/projects/{project_id}/files/{relpath:path}")
    def put_file(project_id: str, relpath: str, body: _FilePut):
        fp = _safe_child(_dir(project_id), relpath)
        ext = fp.suffix.lower()
        if ext not in _PUT_FILE_EXTS:
            raise HTTPException(400, f"refusing to write {ext or '<no extension>'} — only "
                                     f"{sorted(_PUT_FILE_EXTS)} are editable through this endpoint")
        content = body.content or ""
        if len(content.encode("utf-8")) > _PUT_FILE_MAX_BYTES:
            raise HTTPException(400, f"file is larger than the {_PUT_FILE_MAX_BYTES}-byte cap")
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, encoding="utf-8")
        return {"ok": True, "path": str(fp)}

    # ---- stage runs ----

    @app.post("/v1/projects/{project_id}/stages/{stage}/run")
    def run_stage(project_id: str, stage: str, body: _RunStage | None = None):
        d = _dir(project_id)
        b = body or _RunStage()
        try:
            meta = mgr.start(d, stage, overrides=b.overrides,
                             smoke=bool(b.smoke), gpu=b.gpu)
        except RuntimeError as e:      # already running
            raise HTTPException(409, str(e))
        except (ValueError, FileNotFoundError) as e:
            raise HTTPException(400, str(e))
        _scan_state["ts"] = 0.0    # next /v1/jobs should see the new job immediately
        return JSONResponse(meta, status_code=201)

    # ---- jobs ----

    @app.get("/v1/jobs")
    def list_jobs(project: str | None = None):
        pd = None
        if project:
            try:
                pd = reg.dir_for(project)
            except KeyError:
                pd = Path(project)
        _scan_jobs()
        return mgr.list(pd)

    @app.get("/v1/jobs/{job_id}")
    def get_job(job_id: str):
        meta = mgr.get(job_id)
        if not meta:
            raise HTTPException(404, f"no job {job_id}")
        meta = dict(meta)
        meta["events"] = mgr.read_events(job_id)[-50:]
        meta["console_tail"] = jobsmod.read_console(Path(meta["dir"]) / "console.log", tail=200)
        return meta

    @app.get("/v1/jobs/{job_id}/log")
    def job_log(job_id: str, tail: int | None = None):
        meta = mgr.get(job_id)
        if not meta:
            raise HTTPException(404, f"no job {job_id}")
        return PlainTextResponse(jobsmod.read_console(Path(meta["dir"]) / "console.log", tail=tail))

    @app.post("/v1/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        try:
            res = mgr.cancel(job_id)
        except KeyError:
            raise HTTPException(404, f"no job {job_id}")
        _scan_state["ts"] = 0.0
        return res

    @app.get("/v1/jobs/{job_id}/events")
    async def job_events(request: Request, job_id: str, since: int = 0):
        meta = mgr.get(job_id)
        if not meta:
            raise HTTPException(404, f"no job {job_id}")

        async def gen():
            seq = since
            saw_terminal = False
            idle = 0
            ended_sent = False
            try:
                while True:
                    if await request.is_disconnected():
                        return    # client tab closed; stop generating, don't send trailers
                    evts = mgr.read_events(job_id, since=seq)
                    for e in evts:
                        seq = e["seq"]
                        if e.get("type") == "stage_end":
                            saw_terminal = True
                        yield f"data: {json.dumps(e)}\n\n"
                    if evts:
                        idle = 0
                    else:
                        idle += 1
                    if saw_terminal:
                        break
                    if not mgr.is_running(job_id) and idle > 4:   # process gone, give the file a moment to flush
                        # one last drain
                        for e in mgr.read_events(job_id, since=seq):
                            seq = e["seq"]
                            yield f"data: {json.dumps(e)}\n\n"
                        break
                    await asyncio.sleep(0.5)
            finally:
                if not ended_sent:
                    yield f"event: end\ndata: {json.dumps({'job_id': job_id})}\n\n"
                    ended_sent = True

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.on_event("shutdown")
    def _kill_children():
        """When the server is asked to shut down (SIGTERM/SIGINT to uvicorn or programmatic stop),
        signal every running stage subprocess so we don't leave orphans. `start_new_session=True`
        means the children are in their own process groups; without this their parent (us) goes
        away and they keep running, attached to PID 1."""
        try:
            n = mgr.kill_all(grace=2.0)
            if n:
                print(f"[serve] shutdown: signalled {n} running stage subprocess(es)")
        except Exception as e:  # noqa: BLE001
            print(f"[serve] shutdown: kill_all failed: {e}", file=sys.stderr)

    # ---- the web UI (static) ----

    _NO_STORE = {"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache"}

    if _WEBUI_DIR.is_dir():
        @app.get("/")
        def index():
            return FileResponse(_WEBUI_DIR / "index.html", headers=_NO_STORE)

        @app.get("/app.js")
        def appjs():
            return FileResponse(_WEBUI_DIR / "app.js", media_type="text/javascript", headers=_NO_STORE)

        @app.get("/style.css")
        def stylecss():
            return FileResponse(_WEBUI_DIR / "style.css", media_type="text/css", headers=_NO_STORE)

        app.mount("/", StaticFiles(directory=str(_WEBUI_DIR), html=True), name="webui")
    else:
        @app.get("/")
        def index_missing():
            return PlainTextResponse("voicepipe server is up. (web UI files not found at "
                                     f"{_WEBUI_DIR}.)  API is under /v1/", status_code=200)

    return app


def _jsonable_default_project() -> Project:
    return Project(name="example")


# --------------------------------------------------------------------------- entrypoint

_MIN_AUTH_TOKEN_LEN = 32   # required for non-loopback binds


def _token_looks_strong(tok: str) -> tuple[bool, str]:
    """Cheap entropy check for the public-exposure path. We don't try to be clever (zxcvbn etc.) —
    just enforce a length floor and rule out the obvious low-entropy mistakes (single-character
    repeats, single-class strings shorter than 32). Returns (ok, reason)."""
    if len(tok) < _MIN_AUTH_TOKEN_LEN:
        return False, f"too short (need at least {_MIN_AUTH_TOKEN_LEN} chars)"
    classes = ((1 if any(c.islower() for c in tok) else 0)
             + (1 if any(c.isupper() for c in tok) else 0)
             + (1 if any(c.isdigit() for c in tok) else 0)
             + (1 if any(not c.isalnum() for c in tok) else 0))
    if classes < 2:
        return False, "low entropy (need at least two character classes — letters + digits or symbols)"
    if len(set(tok)) < 8:
        return False, "low entropy (fewer than 8 distinct characters)"
    return True, ""


def main(argv=None):
    # Load the env file BEFORE argparse so --auth-token can take its default from
    # $VOICEPIPE_AUTH_TOKEN as written in ~/.config/voicepipe/env. (The argparse `default=`
    # expression is evaluated at parse time; we previously ran the file-load after that, which
    # silently lost env-file tokens whose value was set there and nowhere else.)
    _load_env_file()
    ap = argparse.ArgumentParser(prog="voicepipe serve", description="run the voicepipe control server + web GUI")
    ap.add_argument("--host", default="127.0.0.1", help="bind host (default 127.0.0.1; use 0.0.0.0 for remote)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--unix-socket", default=None, help="bind a Unix domain socket instead of host:port (no auth; for the native app)")
    ap.add_argument("--auth-token", default=os.environ.get("VOICEPIPE_AUTH_TOKEN"),
                    help="shared password for /v1/* over TCP (also $VOICEPIPE_AUTH_TOKEN). Required automatically for non-loopback hosts.")
    ap.add_argument("--no-auth", action="store_true",
                    help="never require auth, even if a token is in the env (the native desktop app passes this; trusted local use only).")
    ap.add_argument("--root", action="append", default=None, help="project root to scan (repeatable). Default: cwd, ~/voicepipe-projects, ./projects, ./scratch")
    ap.add_argument("--no-browser", action="store_true", help="don't try to open a browser")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])
    _start_parent_watchdog()                           # exit if the desktop-shell parent dies (if VOICEPIPE_PARENT_PID set)
    if args.no_auth:
        args.auth_token = None                         # the native app passes --no-auth; ignore any ambient token

    _seed_registry_from_repo()                          # dev checkout: keep projects/ and scratch/ visible (no-op when packaged)
    roots = _resolve_roots(args.root)

    over_uds = bool(args.unix_socket)
    is_loopback = args.host in ("127.0.0.1", "::1", "localhost")
    auth_required = (not over_uds) and (not args.no_auth) and ((not is_loopback) or bool(args.auth_token))
    if (not over_uds) and (not is_loopback) and (not args.auth_token) and (not args.no_auth):
        raise SystemExit("binding a non-loopback host requires --auth-token (or $VOICEPIPE_AUTH_TOKEN), "
                         "or pass --no-auth if you really mean to expose it unauthenticated")
    # When we're putting auth in front of network-reachable traffic, the token has to actually be
    # a token. Refuse short/low-entropy values rather than silently rubber-stamping them.
    if (not over_uds) and (not is_loopback) and args.auth_token:
        ok, why = _token_looks_strong(args.auth_token)
        if not ok:
            raise SystemExit(f"--auth-token is {why}. Use at least {_MIN_AUTH_TOKEN_LEN} chars "
                             f"with mixed character classes — e.g. `python -c \"import secrets; print(secrets.token_urlsafe(32))\"`.")
    if auth_required:
        print("[serve] auth ENABLED — clients must present the token")
    elif over_uds:
        print(f"[serve] Unix socket {args.unix_socket} — auth bypassed (trusted local app)")
    elif args.no_auth:
        print("[serve] --no-auth — auth disabled")

    app = create_app(roots, auth_token=args.auth_token, auth_required=auth_required,
                     bind_host=None if over_uds else args.host)

    try:
        import uvicorn
    except ImportError:
        raise SystemExit("uvicorn not installed — `pip install -e .[gui]`")

    # Use pure-Python uvicorn internals (asyncio loop, h11 parser) — negligible perf cost for a
    # localhost control server, and it sidesteps the uvloop/httptools C-extension headaches when
    # this is run from a PyInstaller-frozen bundle (the desktop sidecar).
    # uvicorn installs its own SIGINT/SIGTERM handlers and runs the FastAPI "shutdown" event
    # before exiting, which is where we kill child stage subprocesses (see _kill_children in
    # create_app). No extra signal handler needed here.
    _uv = dict(log_level="info", loop="asyncio", http="h11")
    if over_uds:
        sock = Path(args.unix_socket).resolve()
        # Ensure the parent directory exists and is mode 0700 — anything in it (incl. the socket
        # we're about to bind) inherits the directory's access control as a baseline. We assume
        # the parent dir is owned by the current user; for a system path like /run/foo/, the
        # system administrator is responsible for that.
        sock.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(sock.parent, 0o700)
        except OSError:
            print(f"[serve] warning: couldn't chmod 0700 {sock.parent} — UDS may be world-accessible",
                  file=sys.stderr)
        # umask 0o077 -> the socket file itself will be created mode 0600. There is a small
        # TOCTOU between unlink and bind; uvicorn handles the bind, and the parent dir's 0700
        # mode is the real access control here.
        os.umask(0o077)
        if sock.exists() or sock.is_symlink():
            try:
                sock.unlink()
            except OSError as e:
                raise SystemExit(f"couldn't remove stale socket {sock}: {e}")
        print(f"[serve] listening on unix:{sock}")
        uvicorn.run(app, uds=str(sock), **_uv)
    else:
        url = f"http://{args.host}:{args.port}"
        print(f"[serve] {url}  (roots: {', '.join(str(r) for r in roots)})")
        if is_loopback and not args.no_browser:
            try:
                import webbrowser, threading
                threading.Timer(1.0, lambda: webbrowser.open(url)).start()
            except Exception:
                pass
        uvicorn.run(app, host=args.host, port=args.port, **_uv)


if __name__ == "__main__":
    main()
