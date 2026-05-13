"""
Propose prompt *categories* for a project from its corpus.

A category is a kind of user message ("casual question", "asks for advice", "personal question",
"challenges the speaker", ...). The synthesis stage samples categories so the dataset isn't all
one shape. This stage shows the corpus to an LLM and asks it to propose a sensible set, weighted.

Reads:  <project>/corpus/*  (a sample of it), <project>/project.toml
Writes: <project>/dataset/categorize/proposed_categories.json
        — with --adopt, also writes the proposal into project.toml's `categories`.

Usage:
  voicepipe categorize --project DIR                 # propose; review the JSON; adopt by hand or:
  voicepipe categorize --project DIR --adopt         # propose and write straight into project.toml
  voicepipe categorize --project DIR --n 14 --model kimi-k2.6
"""

import argparse
import json
import re
import sys
import time
import traceback
from pathlib import Path

from pipeline import events
from pipeline.providers import get_chat_provider

_PROMPT = """You are designing a fine-tuning dataset that teaches a model to talk in the voice
of a particular character (described below). The dataset is built from synthesized
(user message, character response) pairs. To keep the user messages varied, we sample from a set
of CATEGORIES — kinds of thing a user might say to this character.

Propose {n} categories. For each: a short snake_case `name`, a one-sentence `description` of what
a user message in that category looks like, and a `weight` (a positive number; they need not sum
to 1, they get normalized). Cover the natural range of how someone would actually talk to this
character — questions, requests, reactions, personal questions, challenges, small talk, requests
to do/explain something — biased toward whatever fits THIS character best given the samples.

Output ONLY a JSON array, like:
[{{"name": "casual_question", "description": "a plain factual or topical question", "weight": 0.3}}, ...]

CHARACTER: {name} — {description}

CORPUS SAMPLES (excerpts):
{samples}
"""


def _gather_samples(proj, max_chars: int = 12000) -> str:
    from pipeline.project import load_corpus
    corpus = load_corpus(proj)
    if not corpus:
        return "(no corpus files found — propose general-purpose categories)"
    out, used = [], 0
    per = max(800, max_chars // max(1, len(corpus)))
    for fname, text in corpus.items():
        chunk = text.strip()[:per]
        out.append(f"--- {fname} ---\n{chunk}")
        used += len(chunk)
        if used >= max_chars:
            break
    return "\n\n".join(out)


def _parse_array(text: str) -> list:
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text).rstrip("`").strip()
    m = re.search(r"\[.*\]", text, re.S)
    if m:
        text = m.group(0)
    try:
        arr = json.loads(text)
    except json.JSONDecodeError:
        return []
    out = []
    for item in arr if isinstance(arr, list) else []:
        if not isinstance(item, dict):
            continue
        name = re.sub(r"[^a-z0-9_]+", "_", str(item.get("name", "")).lower()).strip("_")
        if not name:
            continue
        try:
            w = float(item.get("weight", 1.0))
        except (TypeError, ValueError):
            w = 1.0
        out.append({"name": name, "description": str(item.get("description", "")).strip(), "weight": max(0.0, w)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="project directory")
    ap.add_argument("--n", type=int, default=12, help="how many categories to propose")
    ap.add_argument("--model", default=None, help="override the proposing model (default: project synthesis.model)")
    ap.add_argument("--adopt", action="store_true", help="write the proposal into project.toml's `categories`")
    args = ap.parse_args()
    events.set_stage("categorize")

    from pipeline.project import load_project
    proj = load_project(args.project)
    model = args.model or proj.synthesis.model
    out_dir = proj.dataset_path("categorize")
    out_path = out_dir / "proposed_categories.json"
    started = time.monotonic()
    events.stage_start(command=[sys.executable, "-m", "pipeline.categorize"] + sys.argv[1:],
                       params={"model": model, "n": args.n, "adopt": bool(args.adopt)},
                       inputs=[str(proj.corpus_path())], outputs=[str(out_path)])
    try:
        events.phase("read_corpus")
        samples = _gather_samples(proj)
        events.phase("propose")
        client = get_chat_provider(proj.synthesis.provider)
        prompt = _PROMPT.format(n=args.n, name=proj.name,
                                description=proj.description or "(no description given)", samples=samples)
        resp = client.chat(model=model, messages=[{"role": "user", "content": prompt}],
                           temperature=0.7, think=(False if proj.synthesis.think is False else None))
        cats = _parse_array(resp)
        out_dir.mkdir(parents=True, exist_ok=True)
        if not cats:
            (out_dir / "_raw_response.txt").write_text(resp, encoding="utf-8")
            events.log("could not parse a category array from the model response — raw saved", level="error")
            events.stage_end(status="error", exit_code=1, duration_sec=time.monotonic() - started,
                             error="no parseable categories in model response")
            print("[categorize] failed to parse a category array; raw response saved.", file=sys.stderr)
            return 1
        out_path.write_text(json.dumps(cats, indent=2), encoding="utf-8")
        print(f"[categorize] proposed {len(cats)} categories -> {out_path}")
        for c in cats:
            print(f"  {c['name']:<26} w={c['weight']:<5} {c['description']}")
        events.artifact(out_path, kind="stats")
        events.progress(current=len(cats), total=len(cats), unit="categories")

        if args.adopt:
            events.phase("adopt")
            _adopt_into_toml(proj.root / "project.toml", cats)
            load_project(proj.root)  # validate
            print(f"[categorize] wrote {len(cats)} categories into {proj.root / 'project.toml'}")
            events.log(f"adopted {len(cats)} categories into project.toml")

        events.stage_end(status="ok", exit_code=0, duration_sec=time.monotonic() - started,
                         summary={"proposed": len(cats), "adopted": bool(args.adopt),
                                  "categories": [c["name"] for c in cats]})
        return 0
    except Exception as e:
        events.stage_end(status="error", exit_code=1, duration_sec=time.monotonic() - started,
                         error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        raise


def _adopt_into_toml(toml_path: Path, cats: list) -> None:
    """Replace the `categories = [...]` array in project.toml (preserving the rest of the file
    textually). If there's no such array, insert one before [[modes]]/[synthesis], else append."""
    text = toml_path.read_text(encoding="utf-8") if toml_path.is_file() else ""
    lines = ["categories = ["]
    for c in cats:
        lines.append(f'  {{ name = "{c["name"]}", description = {json.dumps(c["description"])}, weight = {c["weight"]} }},')
    lines.append("]")
    block = "\n".join(lines)
    pat_multi = re.compile(r"^categories\s*=\s*\[.*?^\]\s*$", re.S | re.M)
    pat_inline = re.compile(r"^categories\s*=\s*\[[^\]]*\]\s*$", re.M)
    if pat_multi.search(text):
        text = pat_multi.sub(block, text, count=1)
    elif pat_inline.search(text):
        text = pat_inline.sub(block, text, count=1)
    else:
        ins = re.search(r"^\[\[modes\]\]|^\[synthesis\]", text, re.M)
        if ins:
            text = text[:ins.start()] + block + "\n\n" + text[ins.start():]
        else:
            text = text.rstrip() + "\n\n" + block + "\n"
    toml_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
