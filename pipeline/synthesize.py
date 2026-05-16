"""
Synthesize (user, response) pairs in a target voice, in diverse batches.

Config from a project (`--project DIR` reads project.toml). Each batch is anchored to one
MODE (a register); modes carry a voice-anchor corpus file, mode-coupled styles, an optional
per-mode category distribution, and optional mode-only context files (e.g. a facts dossier).
The synthesis system message is assembled as: synth_preamble + variety_menus + mode block +
content_rules + glossary.

Reads:  <project>/project.toml, <project>/corpus/*, <project>/<glossary_file>, <project>/<seeds_file>
Writes: <project>/dataset/raw/batch_NNNNN.jsonl  (+ _manifest.jsonl, + _raw_debug/ for under-parsed batches)
Re-runnable: counts existing batches and continues toward `synthesis.target`.

Usage:
  python -m pipeline.synthesize --project scratch/dec-bot --target 30 --pairs-per-batch 10   # smoke
  python -m pipeline.synthesize --project scratch/dec-bot                                     # full (target from config)
  python -m pipeline.synthesize --project scratch/dec-bot --mode BIOGRAPHICAL                 # force one mode
"""

import argparse
import json
import random
import re
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait
from datetime import datetime, timezone
from pathlib import Path

from pipeline import events
from pipeline.providers import get_chat_provider


def weighted_choice(items):
    """items: list of (value, weight)."""
    total = sum(w for _, w in items)
    if total <= 0:
        return items[0][0]
    r = random.random() * total
    upto = 0.0
    for value, w in items:
        upto += w
        if upto >= r:
            return value
    return items[-1][0]


# ---------- response parsing (robust to several malformed shapes the synth model emits) ----------


def _is_valid_pair(obj) -> bool:
    return (
        isinstance(obj, dict)
        and "messages" in obj
        and isinstance(obj["messages"], list)
        and len(obj["messages"]) == 2
        and obj["messages"][0].get("role") == "user"
        and obj["messages"][1].get("role") == "assistant"
        and isinstance(obj["messages"][0].get("content"), str)
        and isinstance(obj["messages"][1].get("content"), str)
        and obj["messages"][0]["content"].strip()
        and obj["messages"][1]["content"].strip()
    )


def _repair_role_drift(line: str) -> str:
    """mistral-large-3 sometimes drifts after the first pair in a batch and writes
       {"role": "assistant": "the content"} instead of the legal
       {"role": "assistant", "content": "the content"}.
    This repair rewrites the malformed shape into the legal one in-place.
    Other shapes pass through unchanged."""
    # Match: "role"  :  "<rolename>"  :  "  -> "role": "<rolename>", "content": "
    return re.sub(
        r'"role"\s*:\s*"(user|assistant|system)"\s*:\s*"',
        r'"role": "\1", "content": "',
        line,
    )


# Tolerant pair extractor — used as a last-resort fallback when strict JSON
# parsing fails completely. Designed for the common Dec-voice failure mode:
# the model writes UNESCAPED double quotes inside content strings ("doctors",
# "patient", "history", etc — Dec's sneering quotation marks). Strict JSON
# parse chokes on those; this regex grabs the text between known landmarks
# without ever asking JSON to validate the inner content.
TOLERANT_PAIR_RE = re.compile(
    r'\{"messages":\s*\[\s*'
    r'\{"role":\s*"user"\s*,\s*"content":\s*"'
    r'(.*?)'                                                 # user content
    r'"\s*\}\s*,\s*'
    r'\{"role":\s*"(?:assistant|content)"\s*[,:]\s*(?:"content":\s*)?"'
    r'(.*?)'                                                 # assistant content
    r'"\s*\}\s*\]\s*\}',
    re.DOTALL,
)


def parse_jsonl_response(text: str) -> list[dict]:
    """Parse generated pairs robustly. The model may return one of:
      (a) Strict JSONL — one pair per line (what the prompt asks for).
      (b) Pretty-printed JSON spanning multiple lines per pair.
      (c) A JSON array wrapping the pairs.
      (d) Any of the above wrapped in markdown fences.
      (e) Drift bug: {"role": "X": "..."} instead of {"role": "X", "content": "..."}.
    Tries each strategy; returns the first one that yields >0 valid pairs."""
    # Strip leading/trailing markdown fences if present.
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
        if cleaned.endswith("```"):
            cleaned = cleaned[: cleaned.rfind("```")].rstrip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].lstrip("\n")

    # Apply the role-drift repair across the whole text — cheap and idempotent.
    cleaned = _repair_role_drift(cleaned)

    pairs: list[dict] = []

    # Strategy 1: strict JSONL (one pair per line).
    for line in cleaned.splitlines():
        line = line.strip().rstrip(",")  # tolerate trailing commas
        if not line or line.startswith("```"):
            continue
        if line.lower().startswith("json"):
            line = line[4:].strip()
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if _is_valid_pair(obj):
            pairs.append(obj)
    # If JSONL caught roughly everything that *looked* like a pair (at least half of the
    # occurrences of `"messages"` — each well-formed pair contains exactly one such key,
    # so `count // 2` is a deliberately loose floor), accept what we got and move on.
    if pairs and len(pairs) >= cleaned.count('"messages"') // 2:
        return pairs
    if pairs:
        # JSONL only got part of them; clear and let strategy 2 try the full text.
        pairs = []

    # Strategy 2: balanced-brace extraction. Walks the text and pulls out each
    # complete top-level JSON object, even if it spans many lines (pretty-printed).
    # Tracks string boundaries so braces inside strings don't confuse depth.
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(cleaned):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    obj = json.loads(cleaned[start : i + 1])
                except json.JSONDecodeError:
                    obj = None
                if obj is not None and _is_valid_pair(obj):
                    pairs.append(obj)
                start = -1
    if pairs:
        return pairs

    # Strategy 3: parse entire response as a JSON array of pairs.
    try:
        arr = json.loads(cleaned)
    except json.JSONDecodeError:
        arr = None
    if isinstance(arr, list):
        for obj in arr:
            if _is_valid_pair(obj):
                pairs.append(obj)
    if pairs:
        return pairs

    # Strategy 4: tolerant regex — recovers pairs with unescaped quotes inside
    # content strings. Dec's sneering quotation marks ("doctors", "history", etc.)
    # routinely defeat strict JSON parsing; this is the workhorse fallback.
    for m in TOLERANT_PAIR_RE.finditer(cleaned):
        user_text = m.group(1)
        asst_text = m.group(2)
        # Decode any properly-escaped sequences the model did get right.
        user_text = user_text.replace('\\"', '"').replace("\\n", "\n").replace("\\t", "\t")
        asst_text = asst_text.replace('\\"', '"').replace("\\n", "\n").replace("\\t", "\t")
        pair = {"messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": asst_text},
        ]}
        if _is_valid_pair(pair):
            pairs.append(pair)
    return pairs




# ---------- prompt assembly ----------

def build_system_message(proj, mode) -> str:
    parts = [proj.synth_preamble.strip()]
    if proj.variety_menus.strip():
        parts.append(proj.variety_menus.strip())
    parts.append(f"MODE FOR THIS BATCH: {mode.name}\n{mode.description.strip()}")
    if proj.content_rules.strip():
        parts.append(proj.content_rules.strip())
    glossary = _glossary_cache.get("text", "")
    if glossary.strip():
        parts.append("GLOSSARY (use these constructions liberally):\n" + glossary.strip())
    return "\n\n".join(parts)


def sample_seeds_for_mode(seeds: list, mode_name: str, k: int = 10) -> list:
    matching = [s for s in seeds if s.get("mode") == mode_name]
    others = [s for s in seeds if s.get("mode") != mode_name]
    n_match = min(8, len(matching), k)
    n_other = min(k - n_match, len(others))
    chosen = random.sample(matching, n_match) + random.sample(others, n_other)
    random.shuffle(chosen)
    return chosen


def build_user_message(proj, corpus: dict, seeds: list, mode, category: str, style: str,
                       length_name: str, length_desc: str, n: int) -> str:
    anchor_name = mode.corpus_anchor
    primary = corpus.get(anchor_name, "")
    # secondary = the OTHER modes' anchor files (alternate-register cross-reference) — NOT this mode's
    # own anchor, and NOT files used only as context dossiers (those are large and belong only in the
    # modes that declare them via context_files).
    all_anchor_names = {m.corpus_anchor for m in proj.modes if m.corpus_anchor}
    secondary = [(name, txt) for name, txt in corpus.items() if name in all_anchor_names and name != anchor_name]

    primary_block = f'<RANT_PRIMARY mode="{mode.name}" source="{anchor_name}">\n{primary}\n</RANT_PRIMARY>'
    secondary_blocks = "\n".join(
        f'<RANT_SECONDARY_REGISTER source="{name}">\n{txt}\n</RANT_SECONDARY_REGISTER>' for name, txt in secondary
    )

    dossier_sections = []
    for cf in (mode.context_files or []):
        txt = corpus.get(cf, "")
        if txt:
            dossier_sections.append(f'<CONTEXT_DOSSIER source="{cf}">\n{txt}\n</CONTEXT_DOSSIER>')
    dossier_block = ("\n\n" + "\n\n".join(dossier_sections) + "\n") if dossier_sections else ""

    sampled = sample_seeds_for_mode(seeds, mode.name, k=10)
    seed_block = "\n".join(json.dumps({"mode": s.get("mode"), "messages": s["messages"]}, ensure_ascii=False) for s in sampled)

    return f"""Generate {n} new (user_message, response) pairs in {mode.name} mode.

CATEGORY: {category}
STYLE EMPHASIS: {style}
LENGTH PROFILE: {length_name} — {length_desc}
MODE: {mode.name}

PRIMARY REFERENCE (the dominant register for this batch — anchor your voice to this one):
{primary_block}

SECONDARY REFERENCES (alternate registers — available for cross-reference but DO NOT dominate the batch):
{secondary_blocks}
{dossier_block}
SEED DIALOGUE EXAMPLES (the character answering a user; most tagged for this batch's mode):
{seed_block}

Now generate {n} NEW pairs in the "{category}" category at the "{length_name}" length, in {mode.name} mode, with stylistic emphasis on: {style}.

CRITICAL DIVERSITY REQUIREMENT: across the {n} pairs your opening shapes MUST vary — do not start every response the same way. Vary how you address the user. Vary your closings.

Each user message should be plausible — a real person might send it. Each response must be in pure {mode.name}-mode voice.

Output STRICT JSONL — one pair per line, no preamble, no postamble, no markdown fences. Format exactly:
{{"messages": [{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}]}}

Generate {n} pairs now."""


# ---------- batch planning / running (rolling-submission concurrency) ----------

_rng_lock = threading.Lock()
_glossary_cache = {}


def _pick_category(proj, mode):
    if mode.category_weights:
        return weighted_choice(list(mode.category_weights.items()))
    if proj.categories:
        return weighted_choice([(c.name, c.weight) for c in proj.categories])
    return "general"


def _pick_style(proj, mode):
    pool = list(proj.synthesis.shared_styles) + list(mode.styles or [])
    return random.choice(pool) if pool else "in-voice"


def _pick_length(proj):
    if not proj.length_profiles:
        return ("medium", "100-250 words per response")
    lp = weighted_choice([(p, p.weight) for p in proj.length_profiles])
    return (lp.name, lp.description)


def make_batch_plan(proj, batch_id: int, mode_cycle_index: int, target_remaining: int, forced_mode=None):
    with _rng_lock:
        if forced_mode:
            mode = proj.mode(forced_mode)
        elif proj.synthesis.balance_modes:
            mode = proj.modes[mode_cycle_index % len(proj.modes)]
        else:
            mode = weighted_choice([(m, m.weight) for m in proj.modes])
        category = _pick_category(proj, mode)
        style = _pick_style(proj, mode)
        length_name, length_desc = _pick_length(proj)
        n = min(proj.synthesis.pairs_per_batch, target_remaining)
        return {"batch_id": batch_id, "mode": mode, "category": category, "style": style,
                "length_name": length_name, "length_desc": length_desc, "n": n}


def run_batch(proj, plan, client, corpus, seeds):
    mode = plan["mode"]
    sys_msg = build_system_message(proj, mode)
    user_msg = build_user_message(proj, corpus, seeds, mode, plan["category"], plan["style"],
                                  plan["length_name"], plan["length_desc"], plan["n"])
    out = {"plan": plan, "pairs": [], "raw": "", "error": None}
    try:
        resp = client.chat(model=proj.synthesis.model,
                           messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
                           temperature=proj.synthesis.temperature, top_p=proj.synthesis.top_p,
                           think=proj.synthesis.think)
        out["raw"] = resp
        out["pairs"] = parse_jsonl_response(resp)
    except Exception as e:
        out["error"] = str(e)
    return out


def write_batch_result(proj, result, out_dir: Path, manifest_path: Path) -> int:
    plan = result["plan"]
    bid, mode, pairs = plan["batch_id"], plan["mode"], result["pairs"]
    style_label = plan["style"].split(":")[0]
    if result["error"]:
        print(f"  [batch {bid}] ERROR: {result['error']}")
        events.log(f"batch {bid} ({mode.name}) failed: {result['error']}", level="error")
        return 0
    if len(pairs) <= plan["n"] // 3:
        dump = out_dir / "_raw_debug" / f"batch_{bid:05d}.txt"
        dump.parent.mkdir(parents=True, exist_ok=True)
        dump.write_text(result["raw"], encoding="utf-8")
        print(f"  [batch {bid}] under-parsed ({len(pairs)}/{plan['n']}) — raw dump saved")
        events.log(f"batch {bid} under-parsed ({len(pairs)}/{plan['n']}) — raw dump saved", level="warn")
        events.artifact(dump, kind="raw_dump")
    if not pairs:
        print(f"  [batch {bid}] WARN no pairs parsed. raw head: {result['raw'][:300]!r}")
        events.log(f"batch {bid} ({mode.name}) parsed 0 pairs", level="warn")
        return 0
    for p in pairs:
        p["mode"] = mode.name
    batch_path = out_dir / f"batch_{bid:05d}.jsonl"
    with batch_path.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    events.artifact(batch_path, kind="dataset", bytes=batch_path.stat().st_size)
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"batch_id": bid, "ts": datetime.now(timezone.utc).isoformat(),
                            "mode": mode.name, "category": plan["category"], "style": style_label,
                            "length": plan["length_name"], "model": proj.synthesis.model,
                            "requested": plan["n"], "got": len(pairs)}) + "\n")
    print(f"  [batch {bid}] mode={mode.name:22s} parsed={len(pairs):>2}/{plan['n']} style={style_label}")
    return len(pairs)


# ---------- driver ----------

def main(args=None):
    if args is None:
        ap = argparse.ArgumentParser()
        ap.add_argument("--project", required=True, help="project directory (reads project.toml)")
        ap.add_argument("--target", type=int, default=None, help="override total-pairs target")
        ap.add_argument("--pairs-per-batch", type=int, default=None)
        ap.add_argument("--model", default=None)
        ap.add_argument("--concurrency", type=int, default=None)
        ap.add_argument("--mode", default=None, help="force every batch to this mode")
        ap.add_argument("--no-balance-modes", action="store_true", help="weighted-random mode selection instead of round-robin")
        ap.add_argument("--temperature", type=float, default=None)
        args = ap.parse_args()
    events.set_stage("synthesize")

    from pipeline.project import load_project, glossary_text, load_corpus, load_seeds
    proj = load_project(args.project)
    # apply CLI overrides onto the synthesis config
    if args.target is not None: proj.synthesis.target = args.target
    if args.pairs_per_batch is not None: proj.synthesis.pairs_per_batch = args.pairs_per_batch
    if args.model: proj.synthesis.model = args.model
    if args.concurrency is not None: proj.synthesis.concurrency = args.concurrency
    if args.temperature is not None: proj.synthesis.temperature = args.temperature
    if args.no_balance_modes: proj.synthesis.balance_modes = False

    if not proj.modes:
        print("[error] project has no modes")
        events.stage_end(status="error", exit_code=2, error="project has no modes"); return 2
    corpus = load_corpus(proj)
    _glossary_cache["text"] = glossary_text(proj)
    seeds = load_seeds(proj)
    for m in proj.modes:
        if m.corpus_anchor and m.corpus_anchor not in corpus:
            msg = f"mode {m.name}: corpus_anchor {m.corpus_anchor!r} not found in corpus/"
            print(f"[error] {msg}")
            events.stage_end(status="error", exit_code=2, error=msg); return 2

    out_dir = proj.dataset_path("raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "_manifest.jsonl"

    existing = sorted(out_dir.glob("batch_*.jsonl"))
    existing_count = sum(sum(1 for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()) for p in existing)
    existing_ids = [int(p.stem.split("_")[1]) for p in existing]
    next_batch_id = (max(existing_ids) + 1) if existing_ids else 0
    mode_cycle_index = next_batch_id
    target_remaining = max(0, proj.synthesis.target - existing_count)
    print(f"[startup] {existing_count} pairs on disk; target_remaining={target_remaining}; "
          f"model={proj.synthesis.model}; concurrency={proj.synthesis.concurrency}; "
          f"{'forced mode '+args.mode if args.mode else ('balance-modes ' + str([m.name for m in proj.modes]) if proj.synthesis.balance_modes else 'weighted-random modes')}")
    events.stage_start(
        command=[sys.executable, "-m", "pipeline.synthesize"] + sys.argv[1:],
        params={"model": proj.synthesis.model, "target": proj.synthesis.target,
                "pairs_per_batch": proj.synthesis.pairs_per_batch, "concurrency": proj.synthesis.concurrency,
                "balance_modes": proj.synthesis.balance_modes, "forced_mode": args.mode,
                "temperature": proj.synthesis.temperature, "modes": [m.name for m in proj.modes]},
        inputs=[str(proj.p(proj.corpus_dir))] + ([str(proj.p(proj.seeds_file))] if proj.seeds_file else []),
        outputs=[str(out_dir)],
        resumed_from={"pairs_on_disk": existing_count, "next_batch_id": next_batch_id} if existing_count else None)
    started_at = time.monotonic()

    client = get_chat_provider(proj.synthesis.provider)
    pairs_done = existing_count
    submitted = 0
    in_flight = set()

    def next_plan():
        nonlocal next_batch_id, mode_cycle_index, target_remaining
        plan = make_batch_plan(proj, next_batch_id, mode_cycle_index, target_remaining, forced_mode=args.mode)
        next_batch_id += 1
        mode_cycle_index += 1
        return plan

    batches_completed = 0
    try:
        with ThreadPoolExecutor(max_workers=proj.synthesis.concurrency) as ex:
            while len(in_flight) < proj.synthesis.concurrency and target_remaining > 0:
                plan = next_plan()
                in_flight.add(ex.submit(run_batch, proj, plan, client, corpus, seeds))
                submitted += 1
                print(f"  [submitted #{submitted}] batch {plan['batch_id']} mode={plan['mode'].name}")
            while in_flight:
                done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                for fut in done:
                    result = fut.result()
                    got = write_batch_result(proj, result, out_dir, manifest_path)
                    pairs_done += got
                    target_remaining -= got
                    batches_completed += 1
                    # Disk count (pairs_done) is allowed to exceed target — slight over-parse is fine
                    # and re-runnable. The progress event payload is capped at target so the UI's
                    # bar doesn't read >100%.
                    pairs_kept_against_target = min(pairs_done, proj.synthesis.target)
                    print(f"  [progress] {pairs_done}/{proj.synthesis.target}")
                    events.progress(current=pairs_kept_against_target, total=proj.synthesis.target, unit="pairs",
                                    detail=f"batch {result['plan']['batch_id']} mode={result['plan']['mode'].name} parsed {got}/{result['plan']['n']}")
                    if target_remaining > 0:
                        plan = next_plan()
                        in_flight.add(ex.submit(run_batch, proj, plan, client, corpus, seeds))
                        submitted += 1
    except Exception as e:
        events.stage_end(status="error", exit_code=1, duration_sec=time.monotonic() - started_at,
                         error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        raise

    print(f"[done] {pairs_done} pairs on disk; submitted {submitted} batches")
    by_mode = {}
    for p in out_dir.glob("batch_*.jsonl"):
        for ln in p.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            try:
                m = json.loads(ln).get("mode", "UNK")
            except json.JSONDecodeError:
                continue
            by_mode[m] = by_mode.get(m, 0) + 1
    events.stage_end(status="ok", exit_code=0, duration_sec=time.monotonic() - started_at,
                     summary={"pairs_total": pairs_done, "batches_completed": batches_completed,
                              "batches_submitted": submitted, "by_mode": by_mode})


if __name__ == "__main__":
    main()
