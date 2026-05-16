"""
Triage deduplicated pairs: an LLM judge scores each on voice fidelity (1-5) and flags policy
violations; pairs scored >= min_keep with no critical flag are kept.

Config from a project (`--project DIR` reads project.toml → TriageConfig: model, rubric,
critical_flags, critical_flag_substrings, batch_size, min_keep, concurrency). Standalone: pass
--in / --out-dir and the CLI flags.

Reads:  <project>/dataset/dedup/pairs.jsonl
Writes: <project>/dataset/triage/scored.jsonl   — every scored pair {pair_id, score, flags, rationale}
        <project>/dataset/triage/keep.jsonl     — the pairs that survive
        <project>/dataset/triage/_audit.jsonl   — per-batch log
Re-runnable: skips already-scored pair_ids in scored.jsonl, then rebuilds keep.jsonl.

Usage:
  python -m pipeline.triage --project scratch/dec-bot
  python -m pipeline.triage --project scratch/dec-bot --min-keep 5 --concurrency 1
"""

import argparse
import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from pipeline import events
from pipeline.providers import get_chat_provider
from pipeline.util import load_jsonl, write_jsonl


def make_is_critical(critical_flags: list, substrings: list):
    crit = {f.lower().strip() for f in critical_flags}
    subs = tuple(s.lower() for s in substrings)
    def is_critical(flag: str) -> bool:
        f = flag.lower().strip()
        return f in crit or any(tok in f for tok in subs)
    return is_critical


def _format_batch(pairs: list, start_idx: int) -> str:
    lines = []
    for i, p in enumerate(pairs):
        lines.append(f"--- PAIR {start_idx + i} ---")
        lines.append(f"USER: {p['messages'][0]['content']}")
        lines.append(f"ASSISTANT: {p['messages'][1]['content']}")
        lines.append("")
    return "\n".join(lines)


def _truncate_partial_tail(path) -> int:
    """If `path` ends mid-line (no trailing newline), trim the trailing partial line in place.
    Returns the number of bytes removed. Safe to call on a non-existent / empty file."""
    from pathlib import Path as _Path
    p = _Path(path)
    if not p.is_file():
        return 0
    size = p.stat().st_size
    if size == 0:
        return 0
    with p.open("rb") as f:
        f.seek(-1, 2)
        last = f.read(1)
    if last == b"\n":
        return 0
    with p.open("rb") as f:
        data = f.read()
    nl = data.rfind(b"\n")
    new_size = nl + 1 if nl >= 0 else 0
    with p.open("rb+") as f:
        f.truncate(new_size)
    removed = size - new_size
    print(f"[resume] trimmed {removed} bytes of partial trailing line from {path}", file=sys.stderr)
    return removed


def _parse(text: str) -> list:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        if line.lower().startswith("json"):
            line = line[4:].strip()
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and "pair_id" in obj and "score" in obj:
                out.append(obj)
        except json.JSONDecodeError:
            continue
    return out


def _process_batch(client, model, rubric, batch_pairs, batch_idxs):
    msgs = [{"role": "system", "content": rubric},
            {"role": "user", "content": _format_batch(batch_pairs, batch_idxs[0])}]
    try:
        response = client.chat(model=model, messages=msgs, temperature=0.0)
    except Exception as e:
        return ([], str(e), batch_idxs)
    return (_parse(response), None, batch_idxs)


def main(args=None):
    if args is None:
        ap = argparse.ArgumentParser()
        ap.add_argument("--project", default=None, help="project directory (reads project.toml → TriageConfig)")
        ap.add_argument("--in", dest="in_path", default=None, help="override input dedup pairs.jsonl")
        ap.add_argument("--out-dir", default=None, help="override output triage dir")
        ap.add_argument("--model", default=None)
        ap.add_argument("--batch-size", type=int, default=None)
        ap.add_argument("--min-keep", type=int, default=None)
        ap.add_argument("--concurrency", type=int, default=None)
        ap.add_argument("--max-pairs", type=int, default=None, help="dry-run cap")
        args = ap.parse_args()
    events.set_stage("triage")

    from pipeline.project import TriageConfig
    if args.project:
        from pipeline.project import load_project
        proj = load_project(args.project)
        tc = proj.triage
        in_path = Path(args.in_path) if args.in_path else proj.dataset_path("dedup", "pairs.jsonl")
        out_dir = Path(args.out_dir) if args.out_dir else proj.dataset_path("triage")
    else:
        tc = TriageConfig()
        in_path = Path(args.in_path or "dataset/dedup/pairs.jsonl")
        out_dir = Path(args.out_dir or "dataset/triage")

    model = args.model or tc.model
    batch_size = args.batch_size or tc.batch_size
    min_keep = args.min_keep if args.min_keep is not None else tc.min_keep
    concurrency = args.concurrency or tc.concurrency
    rubric = tc.rubric
    if not rubric:
        print("[error] no triage rubric configured (project [triage].rubric)", flush=True)
        events.stage_end(status="error", exit_code=2, error="no triage rubric configured (project [triage].rubric)")
        return 2
    is_critical = make_is_critical(tc.critical_flags, tc.critical_flag_substrings)

    out_dir.mkdir(parents=True, exist_ok=True)
    scored_path, keep_path, audit_path = out_dir / "scored.jsonl", out_dir / "keep.jsonl", out_dir / "_audit.jsonl"

    pairs = load_jsonl(in_path)
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    print(f"[load] {len(pairs)} pairs to triage")

    done_ids = set()
    if scored_path.exists():
        # If a previous run was SIGKILLed mid-append, the file may end in a partial line.
        # Trim that tail before reopening for append so we don't concatenate a half-line
        # with the next write.
        _truncate_partial_tail(scored_path)
        for s in load_jsonl(scored_path):
            if "pair_id" in s:
                done_ids.add(s["pair_id"])
        print(f"[resume] {len(done_ids)} pairs already scored")

    batches, cur, cur_idx = [], [], []
    for i, p in enumerate(pairs):
        if i in done_ids:
            continue
        cur.append(p); cur_idx.append(i)
        if len(cur) >= batch_size:
            batches.append((cur, cur_idx)); cur, cur_idx = [], []
    if cur:
        batches.append((cur, cur_idx))

    n_pairs = len(pairs)
    total_batches = (n_pairs + batch_size - 1) // batch_size
    already_done_batches = total_batches - len(batches)
    events.stage_start(command=[sys.executable, "-m", "pipeline.triage"] + sys.argv[1:],
                       params={"model": model, "batch_size": batch_size, "min_keep": min_keep,
                               "concurrency": concurrency},
                       inputs=[str(in_path)], outputs=[str(keep_path), str(scored_path)],
                       resumed_from={"scored_pairs": len(done_ids)} if done_ids else None)
    started_at = time.monotonic()
    progress_done = already_done_batches
    if progress_done:
        events.progress(current=progress_done, total=total_batches, unit="batches", detail="resumed")

    try:
        if not batches:
            print("[triage] nothing to do — all pairs already scored")
        else:
            print(f"[triage] {len(batches)} batches @ concurrency={concurrency} model={model}")
            client = get_chat_provider(tc.provider)
            lock = threading.Lock()
            completed = total_scored = 0
            with scored_path.open("a", encoding="utf-8") as sf, audit_path.open("a", encoding="utf-8") as af, \
                 ThreadPoolExecutor(max_workers=concurrency) as ex:
                futs = {ex.submit(_process_batch, client, model, rubric, bp, bi): (bp, bi) for bp, bi in batches}
                for fut in as_completed(futs):
                    bp, bi = futs[fut]
                    scores, err, _ = fut.result()
                    completed += 1
                    with lock:
                        if err:
                            print(f"[{completed}/{len(batches)}] batch@{bi[0]:>4}: ERROR {err[:120]}")
                            events.log(f"triage batch @{bi[0]} failed: {err[:200]}", level="error")
                        else:
                            total_scored += len(scores)
                            print(f"[{completed}/{len(batches)}] batch@{bi[0]:>4}: parsed {len(scores)}/{len(bp)}")
                            for s in scores:
                                sf.write(json.dumps(s) + "\n")
                                sf.flush()
                                os.fsync(sf.fileno())
                            af.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(),
                                                 "batch_size": len(bp), "scored": len(scores), "first_idx": bi[0]}) + "\n")
                            af.flush()
                            os.fsync(af.fileno())
                        events.progress(current=progress_done + completed, total=total_batches, unit="batches",
                                        detail=f"batch @{bi[0]} parsed {len(scores)}/{len(bp)}")
            print(f"[triage] done. {total_scored} new scores across {completed} batches.")

        # finalize keep.jsonl
        print("[finalize] building keep.jsonl")
        events.phase("finalize")
        scores_by_id = {}
        for s in load_jsonl(scored_path):
            if "pair_id" in s:
                scores_by_id[s["pair_id"]] = s
        kept = dropped_low = dropped_flag = 0
        rows = []
        for idx, p in enumerate(pairs):
            s = scores_by_id.get(idx)
            if s is None:
                continue
            if s.get("score", 0) < min_keep:
                dropped_low += 1; continue
            if any(is_critical(fl) for fl in s.get("flags", [])):
                dropped_flag += 1; continue
            rows.append(p); kept += 1
        write_jsonl(keep_path, rows)
        print(f"[finalize] kept={kept} dropped_low_score={dropped_low} dropped_flagged={dropped_flag}")
        events.artifact(keep_path, kind="dataset", bytes=Path(keep_path).stat().st_size)
        events.stage_end(status="ok", exit_code=0, duration_sec=time.monotonic() - started_at,
                         summary={"scored": len(scores_by_id), "kept": kept,
                                  "dropped_low": dropped_low, "dropped_flagged": dropped_flag})
    except Exception as e:
        events.stage_end(status="error", exit_code=1, duration_sec=time.monotonic() - started_at,
                         error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()
