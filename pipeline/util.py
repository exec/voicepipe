"""Small shared helpers used across pipeline stages."""

import json
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


def load_env_file() -> None:
    """Load `KEY=VALUE` lines from ~/.config/voicepipe/env into os.environ (without clobbering
    anything already set). The GUI's `voicepipe serve` and every `voicepipe <stage>` CLI command
    call this, so OLLAMA_API_KEY / VOICEPIPE_AUTH_TOKEN configured once in that file (or via the
    GUI Settings) just work. Same file `deploy/voicepipe.service` reads via EnvironmentFile=."""
    import os
    from pathlib import Path
    cfg = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "voicepipe" / "env"
    if not cfg.is_file():
        return
    for line in cfg.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v
