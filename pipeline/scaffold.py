"""
Scaffold a new project directory from a bundled template.

Used by `voicepipe new <name>` (CLI) and `POST /v1/projects` (the GUI server). A template is
just a directory under pipeline/templates/ containing a project.toml (with {{NAME}} / {{DESCRIPTION}}
placeholders) plus the prose-block files, an empty corpus/, and a seeds file.
"""

import shutil
from pathlib import Path

_TEMPLATES_DIR = Path(__file__).parent / "templates"

# Human-facing one-liners for the bundled templates (keyed by directory name).
_TEMPLATE_DESCRIPTIONS = {
    "character": "Fleshed-out single-voice character: synthesis preamble, variety menus, content "
                 "rules, a triage rubric, a deploy system prompt, one VOICE mode, sensible "
                 "hyperparameters. Add your corpus + a few seed pairs and run.",
    "blank": "Bare minimum — just name/description and one empty VOICE mode. Everything else "
             "falls back to defaults; fill in prose blocks and modes yourself.",
}

# File types we run {{...}} placeholder substitution on (others are copied verbatim).
_SUBST_SUFFIXES = {".toml", ".md", ".txt", ".jsonl", ".cfg", ".ini", ""}


def list_templates() -> list[dict]:
    """[{id, description}] for every bundled template."""
    out = []
    if not _TEMPLATES_DIR.is_dir():
        return out
    for d in sorted(_TEMPLATES_DIR.iterdir()):
        if d.is_dir() and (d / "project.toml").is_file():
            out.append({"id": d.name, "description": _TEMPLATE_DESCRIPTIONS.get(d.name, "")})
    return out


def template_dir(template: str) -> Path:
    src = _TEMPLATES_DIR / template
    if not (src / "project.toml").is_file():
        raise ValueError(f"unknown template {template!r}; have: {[t['id'] for t in list_templates()]}")
    return src


def create_project(dest, *, template: str = "character", name: str, description: str = "") -> Path:
    """Copy `template` to `dest`, substituting {{NAME}}/{{DESCRIPTION}}. Returns the project dir.
    `dest` must not exist or must be empty (and must not already contain a project.toml)."""
    dest = Path(dest).expanduser().resolve()
    if dest.exists() and dest.is_file():
        raise FileExistsError(f"{dest} is a file")
    if dest.is_dir() and (dest / "project.toml").is_file():
        raise FileExistsError(f"{dest} already contains a project.toml; refusing to overwrite")
    if dest.is_dir() and any(dest.iterdir()):
        raise FileExistsError(f"{dest} exists and is not empty")
    src = template_dir(template)

    if dest.exists():
        # copy into the existing empty dir
        for item in src.iterdir():
            target = dest / item.name
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)
    else:
        shutil.copytree(src, dest)

    repl = {"{{NAME}}": name, "{{DESCRIPTION}}": description or f"A character voice: {name}."}
    for fp in dest.rglob("*"):
        if not fp.is_file() or fp.suffix.lower() not in _SUBST_SUFFIXES:
            continue
        try:
            txt = fp.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        new = txt
        for k, v in repl.items():
            new = new.replace(k, v)
        if new != txt:
            fp.write_text(new, encoding="utf-8")

    (dest / "dataset").mkdir(exist_ok=True)
    return dest
