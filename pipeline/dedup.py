"""
Dedup the raw synthesis output: normalize typography, drop exact dupes, drop near-dupes via
cosine similarity on local embeddings, and cap over-saturated phrases ("closers") so no single
phrase dominates the dataset.

Config from a project (`--project DIR` reads project.toml → DedupConfig); paths resolve relative
to the project. Standalone use: pass `--raw-dir` / `--out` and the CLI flags.

Reads:  <project>/dataset/raw/batch_*.jsonl  (+ _manifest.jsonl for category lookup)
Writes: <project>/dataset/dedup/pairs.jsonl

Steps: load+normalize → hash dedup → cosine dedup (local Ollama embeddings) → closer caps.

Usage:
  ollama pull nomic-embed-text                                  # one-time, locally
  python -m pipeline.dedup --project scratch/dec-bot
  python -m pipeline.dedup --project scratch/dec-bot --threshold 0.90 --no-downsample
  python -m pipeline.dedup --project scratch/dec-bot --skip-embed     # hash-only (no local Ollama)
"""

import argparse
import hashlib
import random
import re
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

from pipeline import events
from pipeline.util import normalize_punctuation, user_text, assistant_text, mode_summary, write_jsonl

# Re-exported for backward compat (other code historically did `from dedup import normalize_punctuation`).
__all__ = ["normalize_punctuation", "main"]

_ABBREV = {"COSMOLOGY": "COS", "PERSECUTION": "PERS", "HISTORICAL_INDICTMENT": "HIST", "BIOGRAPHICAL": "BIO"}


# ---------- loaders ----------

def _load_manifest(raw_dir: Path) -> dict:
    mp = raw_dir / "_manifest.jsonl"
    if not mp.exists():
        return {}
    out = {}
    for line in mp.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                import json
                m = json.loads(line)
                out[m["batch_id"]] = m
            except Exception:
                pass
    return out


def load_raw_pairs(raw_dir: Path) -> list:
    """Load + normalize all raw pairs; backfill (mode, category) from the manifest; drop malformed."""
    import json
    manifest = _load_manifest(raw_dir)
    pairs, malformed = [], 0
    for batch_path in sorted(raw_dir.glob("batch_*.jsonl")):
        try:
            batch_id = int(batch_path.stem.split("_")[1])
        except (ValueError, IndexError):
            batch_id = None
        meta = manifest.get(batch_id, {})
        for line in batch_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                pair = json.loads(line)
            except json.JSONDecodeError:
                continue
            msgs = pair.get("messages")
            if (not isinstance(msgs, list) or len(msgs) != 2
                    or not isinstance(msgs[0].get("content"), str) or not isinstance(msgs[1].get("content"), str)
                    or not msgs[0]["content"].strip() or not msgs[1]["content"].strip()):
                malformed += 1
                continue
            for msg in msgs:
                msg["content"] = normalize_punctuation(msg["content"])
            if "mode" not in pair and meta.get("mode"):
                pair["mode"] = meta["mode"]
            if meta.get("category") and "category" not in pair:
                pair["category"] = meta["category"]
            pairs.append(pair)
    if malformed:
        print(f"[load] dropped {malformed} malformed pair(s)")
    return pairs


# ---------- dedup steps ----------

def hash_dedup(pairs: list) -> list:
    seen, out = set(), []
    for p in pairs:
        key = hashlib.sha256((user_text(p).strip().lower() + "|||" + assistant_text(p).strip().lower()).encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _embed(texts: list, model: str, base_url: str) -> list:
    import requests
    url = base_url.rstrip("/") + "/api/embed"
    out = []
    backoffs = (1.0, 3.0, 9.0)  # 3 attempts: immediate, then 1s, 3s, 9s waits before each retry
    for i in range(0, len(texts), 64):
        batch = texts[i:i + 64]
        last_err: Exception | None = None
        for attempt in range(len(backoffs) + 1):
            try:
                r = requests.post(url, json={"model": model, "input": batch}, timeout=180)
                r.raise_for_status()
                out.extend(r.json()["embeddings"])
                last_err = None
                break
            except Exception as e:
                last_err = e
                if attempt < len(backoffs):
                    wait = backoffs[attempt]
                    print(f"[embed] batch {i}-{i+len(batch)} attempt {attempt+1} failed ({e}); "
                          f"retrying in {wait}s")
                    time.sleep(wait)
        if last_err is not None:
            raise last_err
    return out


def cosine_dedup(pairs: list, threshold: float, embed_model: str, embed_base_url: str) -> list:
    try:
        import numpy as np
    except ImportError:
        print("[warn] numpy not installed; skipping cosine dedup")
        return pairs
    texts = [assistant_text(p) for p in pairs]
    print(f"[embed] {len(texts)} assistant texts via {embed_model}")
    try:
        embs = _embed(texts, embed_model, embed_base_url)
    except Exception as e:
        # loud: cosine dedup is silently disabled when the embed endpoint is unreachable; the
        # rest of the pipeline still runs but the output will contain near-duplicates.
        msg = f"embedding endpoint failed after retries ({e}); SKIPPING cosine dedup — output will retain near-duplicates"
        print(f"[ERROR] {msg}", file=sys.stderr)
        events.log(msg, level="error")
        return pairs
    embs = np.array(embs, dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs = embs / norms
    # TODO: O(N^2) with per-iteration np.stack — fine up to ~20k pairs; for larger runs,
    # batch with FAISS or compute a Jaccard pre-filter to shrink candidate set.
    kept_idx, kept_embs = [], []
    for i in range(len(pairs)):
        if not kept_embs:
            kept_idx.append(i); kept_embs.append(embs[i]); continue
        if np.dot(np.stack(kept_embs), embs[i]).max() >= threshold:
            continue
        kept_idx.append(i); kept_embs.append(embs[i])
    return [pairs[i] for i in kept_idx]


# ---------- closer caps ----------

def downsample_closers(pairs: list, closers: list, seed: int = 42) -> list:
    """`closers` is a list of (name, compiled_regex, cap_fraction). For each, if more than `cap`
    of the dataset matches in the last 200 chars, drop randomly from over-represented (mode, category)
    buckets until at cap."""
    rng = random.Random(seed)
    total = len(pairs)
    if total == 0:
        return pairs
    print(f"[downsample] starting with {total} pairs")
    surviving = set(range(total))
    for name, pat, cap in closers:
        with_closer = [i for i in surviving if pat.search(assistant_text(pairs[i])[-200:])]
        target = int(round(cap * total))
        before = len(with_closer)
        if before <= target:
            print(f"  {name}: {before}/{total} = {100*before/total:.1f}% (cap {100*cap:.0f}%) — under cap")
            continue
        n_to_drop = before - target
        buckets = defaultdict(list)
        for i in with_closer:
            buckets[(pairs[i].get("mode", "UNK"), pairs[i].get("category", "UNK"))].append(i)
        for v in buckets.values():
            rng.shuffle(v)
        dropped = 0
        while dropped < n_to_drop:
            biggest = max(buckets, key=lambda k: len(buckets[k]))
            if not buckets[biggest]:
                break
            surviving.discard(buckets[biggest].pop())
            dropped += 1
        after = sum(1 for i in surviving if pat.search(assistant_text(pairs[i])[-200:]))
        print(f"  {name}: {before} -> {after} ({100*after/total:.1f}%, target {100*cap:.0f}%, dropped {dropped})")
    print(f"[downsample] {len(surviving)} pairs remain after closer caps")
    return [pairs[i] for i in sorted(surviving)]


# ---------- driver ----------

def main(args=None):
    if args is None:
        ap = argparse.ArgumentParser()
        ap.add_argument("--project", default=None, help="project directory (reads project.toml → DedupConfig)")
        ap.add_argument("--raw-dir", default=None, help="override input dir of batch_*.jsonl")
        ap.add_argument("--out", default=None, help="override output pairs.jsonl path")
        ap.add_argument("--threshold", type=float, default=None, help="override cosine threshold")
        ap.add_argument("--skip-embed", action="store_true", help="hash-only dedup")
        ap.add_argument("--no-downsample", action="store_true", help="skip closer caps")
        args = ap.parse_args()
    events.set_stage("dedup")

    from pipeline.project import DedupConfig
    if args.project:
        from pipeline.project import load_project
        proj = load_project(args.project)
        dc = proj.dedup
        raw_dir = Path(args.raw_dir) if args.raw_dir else proj.dataset_path("raw")
        out_path = Path(args.out) if args.out else proj.dataset_path("dedup", "pairs.jsonl")
    else:
        dc = DedupConfig()
        raw_dir = Path(args.raw_dir or "dataset/raw")
        out_path = Path(args.out or "dataset/dedup/pairs.jsonl")

    threshold = args.threshold if args.threshold is not None else dc.cosine_threshold
    skip_embed = args.skip_embed or dc.skip_embed
    closers = [(c.name, re.compile(c.regex, re.I), c.cap) for c in dc.closer_patterns]

    started_at = time.monotonic()
    try:
        pairs = load_raw_pairs(raw_dir)
        n_in = len(pairs)
        print(f"[load]               {len(pairs):>5} pairs   {mode_summary(pairs, _ABBREV)}")
        events.stage_start(command=[sys.executable, "-m", "pipeline.dedup"] + sys.argv[1:],
                           params={"cosine_threshold": threshold, "skip_embed": skip_embed,
                                   "embed_model": dc.embed_model, "downsample": not args.no_downsample,
                                   "closer_patterns": [c.name for c in dc.closer_patterns]},
                           inputs=[str(raw_dir)], outputs=[str(out_path)])
        events.progress(current=n_in, total=n_in, unit="pairs", detail="loaded")

        events.phase("hash_dedup")
        pairs = hash_dedup(pairs)
        n_hash = len(pairs)
        print(f"[hash dedup]         {len(pairs):>5} pairs   {mode_summary(pairs, _ABBREV)}")
        events.progress(current=n_hash, total=n_in, unit="pairs", detail="after hash dedup")

        n_cosine = None
        if not skip_embed:
            events.phase("cosine_dedup")
            pairs = cosine_dedup(pairs, threshold, dc.embed_model, dc.embed_base_url)
            n_cosine = len(pairs)
            print(f"[cosine dedup @{threshold:.2f}] {len(pairs):>5} pairs   {mode_summary(pairs, _ABBREV)}")
            events.progress(current=n_cosine, total=n_in, unit="pairs", detail=f"after cosine dedup @{threshold:.2f}")

        n_down = None
        if not args.no_downsample and closers:
            events.phase("downsample")
            pairs = downsample_closers(pairs, closers)
            n_down = len(pairs)
            print(f"[after downsample]   {len(pairs):>5} pairs   {mode_summary(pairs, _ABBREV)}")
            events.progress(current=n_down, total=n_in, unit="pairs", detail="after closer caps")

        write_jsonl(out_path, pairs)
        print(f"[write]              {len(pairs):>5} pairs   {out_path}")
        events.artifact(out_path, kind="dataset", bytes=Path(out_path).stat().st_size)
        events.stage_end(status="ok", exit_code=0, duration_sec=time.monotonic() - started_at,
                         summary={"in": n_in, "after_hash": n_hash, "after_cosine": n_cosine,
                                  "after_downsample": n_down, "out": len(pairs)})
    except Exception as e:
        events.stage_end(status="error", exit_code=1, duration_sec=time.monotonic() - started_at,
                         error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()
