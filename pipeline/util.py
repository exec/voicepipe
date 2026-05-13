"""Small shared helpers used across pipeline stages."""

import json
import os
from collections import Counter
from pathlib import Path


def normalize_punctuation(text: str) -> str:
    """Strip LLM autocorrect typography to plain ASCII (em/en dashes, curly quotes, ellipsis)."""
    return (text
            .replace("—", "--").replace("–", "-")
            .replace("’", "'").replace("‘", "'")
            .replace("“", '"').replace("”", '"')
            .replace("…", "..."))


def load_jsonl(path) -> list:
    path = Path(path)
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def write_jsonl(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def user_text(pair: dict) -> str:
    return pair["messages"][0]["content"]


def assistant_text(pair: dict) -> str:
    return pair["messages"][1]["content"]


def valid_pair(obj) -> bool:
    msgs = obj.get("messages") if isinstance(obj, dict) else None
    return (isinstance(msgs, list) and len(msgs) == 2
            and isinstance(msgs[0].get("content"), str) and isinstance(msgs[1].get("content"), str)
            and msgs[0]["content"].strip() and msgs[1]["content"].strip())


def mode_summary(pairs: list, abbrev: dict | None = None) -> str:
    """Compact 'MODE=N MODE=N' summary for stage-by-stage logging."""
    abbrev = abbrev or {}
    c = Counter(p.get("mode", "UNK") for p in pairs)
    return " ".join(f"{abbrev.get(m, m)}={n}" for m, n in sorted(c.items()))


# ---- env file (canonical loader + writer) --------------------------------------------
# ~/.config/voicepipe/env is the single place OLLAMA_API_KEY / VOICEPIPE_AUTH_TOKEN etc.
# live for both `voicepipe serve` and the stage CLIs (and deploy/voicepipe.service via
# EnvironmentFile=). Quoting convention: values may be wrapped in matching single or
# double quotes on read; values written back are wrapped in double quotes when they
# contain whitespace, an equals sign, or a quote — so write -> read is a round trip.

def env_file_path() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "voicepipe" / "env"


def _unquote(v: str) -> str:
    s = v.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        inner = s[1:-1]
        if s[0] == '"':
            # symmetric with _quote_for_write: only "\\" and "\"" are escapable
            out, i = [], 0
            while i < len(inner):
                if inner[i] == "\\" and i + 1 < len(inner) and inner[i + 1] in ('"', "\\"):
                    out.append(inner[i + 1]); i += 2
                else:
                    out.append(inner[i]); i += 1
            return "".join(out)
        return inner
    return s


def _quote_for_write(v: str) -> str:
    if v == "" or any(c in v for c in (' ', '\t', '"', "'", '=', '#')):
        return '"' + v.replace('\\', '\\\\').replace('"', '\\"') + '"'
    return v


def load_env_file() -> None:
    """Load `KEY=VALUE` lines from ~/.config/voicepipe/env into os.environ (without clobbering
    anything already set). The GUI's `voicepipe serve` and every `voicepipe <stage>` CLI command
    call this, so OLLAMA_API_KEY / VOICEPIPE_AUTH_TOKEN configured once in that file (or via the
    GUI Settings) just work. Same file `deploy/voicepipe.service` reads via EnvironmentFile=."""
    cfg = env_file_path()
    if not cfg.is_file():
        return
    for line in cfg.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), _unquote(v)
        if k and k not in os.environ:
            os.environ[k] = v


def set_env_file_var(key: str, value: str) -> None:
    """Update or append `KEY=value` in ~/.config/voicepipe/env (preserving other lines/comments),
    and set os.environ[key] so it takes effect for stage subprocesses spawned afterward. An empty
    value removes the key."""
    cfg = env_file_path()
    cfg.parent.mkdir(parents=True, exist_ok=True)
    lines = cfg.read_text(encoding="utf-8").splitlines() if cfg.is_file() else [
        "# voicepipe — loaded by `voicepipe serve` (and read by deploy/voicepipe.service via EnvironmentFile=)"
    ]
    out, found = [], False
    for ln in lines:
        s = ln.strip()
        if s and not s.startswith("#") and s.split("=", 1)[0].strip() == key:
            found = True
            if value:
                out.append(f"{key}={_quote_for_write(value)}")
            # else: drop the line
        else:
            out.append(ln)
    if not found and value:
        out.append(f"{key}={_quote_for_write(value)}")
    cfg.write_text("\n".join(out) + "\n", encoding="utf-8")
    # 0o600 is mandatory for secrets — refuse to leave it world-readable.
    try:
        os.chmod(cfg, 0o600)
    except OSError as e:
        raise RuntimeError(f"failed to chmod 0600 on {cfg}: {e}") from e
    if value:
        os.environ[key] = value
    else:
        os.environ.pop(key, None)
