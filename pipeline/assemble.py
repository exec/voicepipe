"""
Assemble the final fine-tuning dataset: combine the triage-kept synthetic pairs + the
hand-written seeds + any salvage pools, normalize typography, shuffle, split into train/val.

Config from a project (`--project DIR` reads project.toml → AssembleConfig + seeds_file +
dataset paths). Standalone: pass --keep / --seeds / --salvage / --out-dir.

Reads:  <project>/dataset/triage/keep.jsonl, <project>/<seeds_file>, <project>/<salvage_paths...>
Writes: <project>/dataset/final/train.jsonl, val.jsonl, stats.json

Usage:
  python -m pipeline.assemble --project scratch/dec-bot
  python -m pipeline.assemble --project scratch/dec-bot --val-fraction 0.05
"""

import argparse
import json
import random
import sys
import time
import traceback
from pathlib import Path

from pipeline import events
from pipeline.util import normalize_punctuation, valid_pair, write_jsonl


def _load(path) -> list:
    """Load jsonl, normalize punctuation on every message turn, keep only well-formed pairs."""
    if not path or not Path(path).is_file():
        return []
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        for m in obj.get("messages", []):
            if isinstance(m.get("content"), str):
                m["content"] = normalize_punctuation(m["content"])
        if valid_pair(obj):
            out.append(obj)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=None, help="project directory (reads project.toml → AssembleConfig)")
    ap.add_argument("--keep", default=None, help="override path to triage keep.jsonl")
    ap.add_argument("--seeds", default=None, help="override path to seeds jsonl")
    ap.add_argument("--salvage", action="append", default=None, help="override salvage jsonl path(s); repeatable")
    ap.add_argument("--out-dir", default=None, help="override output dir")
    ap.add_argument("--val-fraction", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    events.set_stage("assemble")

    from pipeline.project import AssembleConfig
    if args.project:
        from pipeline.project import load_project
        proj = load_project(args.project)
        ac = proj.assemble
        keep_path = Path(args.keep) if args.keep else proj.dataset_path("triage", "keep.jsonl")
        seeds_path = Path(args.seeds) if args.seeds else (proj.p(proj.seeds_file) if proj.seeds_file else None)
        salvage_paths = ([Path(s) for s in args.salvage] if args.salvage else [proj.p(s) for s in ac.salvage_paths])
        out_dir = Path(args.out_dir) if args.out_dir else proj.dataset_path("final")
    else:
        ac = AssembleConfig()
        keep_path = Path(args.keep or "dataset/triage/keep.jsonl")
        seeds_path = Path(args.seeds) if args.seeds else None
        salvage_paths = [Path(s) for s in (args.salvage or [])]
        out_dir = Path(args.out_dir or "dataset/final")

    val_fraction = args.val_fraction if args.val_fraction is not None else ac.val_fraction
    seed = args.seed if args.seed is not None else ac.seed

    started_at = time.monotonic()
    events.stage_start(command=[sys.executable, "-m", "pipeline.assemble"] + sys.argv[1:],
                       params={"val_fraction": val_fraction, "seed": seed,
                               "salvage_paths": [str(p) for p in salvage_paths]},
                       inputs=[str(p) for p in ([keep_path, seeds_path] + salvage_paths) if p],
                       outputs=[str(out_dir / "train.jsonl"), str(out_dir / "val.jsonl")])
    try:
        seeds = _load(seeds_path)
        keeps = _load(keep_path)
        salvage = []
        for sp in salvage_paths:
            salvage += _load(sp)
        print(f"[load] {len(seeds)} seeds, {len(keeps)} curated synthetic, {len(salvage)} salvage")
    except Exception as e:
        events.stage_end(status="error", exit_code=1, duration_sec=time.monotonic() - started_at,
                         error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        raise

    random.seed(seed)
    all_pairs = seeds + keeps + salvage
    random.shuffle(all_pairs)
    n_val = max(50, int(len(all_pairs) * val_fraction)) if all_pairs else 0
    val, train = all_pairs[:n_val], all_pairs[n_val:]

    def lens(rows):
        return [len(p["messages"][1]["content"].split()) for p in rows]
    tl, vl = lens(train), lens(val)
    stats = {
        "n_seeds": len(seeds), "n_curated_synthetic": len(keeps), "n_salvage": len(salvage),
        "n_train": len(train), "n_val": len(val),
        "train_response_words": {"min": min(tl) if tl else 0, "max": max(tl) if tl else 0,
                                 "mean": sum(tl) / len(tl) if tl else 0,
                                 "median": sorted(tl)[len(tl) // 2] if tl else 0},
        "val_response_words": {"min": min(vl) if vl else 0, "max": max(vl) if vl else 0,
                               "mean": sum(vl) / len(vl) if vl else 0},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[write] {out_dir}/train.jsonl ({len(train)}), val.jsonl ({len(val)}), stats.json")
    print(json.dumps(stats, indent=2))
    events.artifact(out_dir / "train.jsonl", kind="dataset")
    events.artifact(out_dir / "val.jsonl", kind="dataset")
    events.artifact(out_dir / "stats.json", kind="stats")
    events.progress(current=len(all_pairs), total=len(all_pairs), unit="rows", detail="assembled")
    events.stage_end(status="ok", exit_code=0, duration_sec=time.monotonic() - started_at, summary=stats)


if __name__ == "__main__":
    main()
