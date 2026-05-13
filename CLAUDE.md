# voicepipe — project guidance

This repo is **voicepipe**: a generalized corpus-to-character pipeline (synthesize a
fine-tuning dataset in a target voice → dedup → triage → assemble → QLoRA-train → deploy via
Ollama). The eventual product is a GUI app that drives this engine with smart defaults.

`scratch/dec-bot/` is the proof-of-concept project — a chatbot in the voice of Francis E. Dec,
Esq. **`scratch/` is .gitignored.** Project-specific guidance for the dec-bot work (the voice
invariants, the slur-strip policy, the v1/v2/v3 history) lives in `scratch/dec-bot/DEC_BOT_NOTES.md`
and `scratch/dec-bot/WRITEUP.md` — read those when working on dec-bot specifically.

## Where things are

- `pipeline/project.py` — the `Project` config schema (the product's data model) + `load_project(dir)`.
  Read this first to understand what a project is. Everything a human would configure is a field
  here with a default.
- `pipeline/<stage>.py` — the pipeline stages, all config-driven (`--project DIR`): synthesize,
  dedup, triage, assemble, train, deploy. (`eval.py`/`infer.py` are dev tools, not yet wired —
  see `pipeline/REFACTOR_STATUS.md`.)
- `pipeline/cli.py` — `python -m pipeline <stage> --project DIR`.
- `scratch/dec-bot/project.toml` — the fully-worked example config. Long prose (synth preamble,
  mode descriptions, content rules, triage rubric, deploy system prompt) lives in
  `scratch/dec-bot/prompts/` and is referenced by filename from the TOML.

## Adding a new character

Make `projects/<name>/` (or `scratch/<name>/` for WIP) with a `project.toml`, a `corpus/`,
optionally `seeds/seed_pairs.jsonl`, and the prose files it references (`prompts/`). Accept the
defaults for everything else. Then: `python -m pipeline synthesize --project projects/<name>` →
`dedup` → `triage` → `assemble` → `train` → `deploy`. See `scratch/dec-bot/project.toml` for the
shape of a fully-specified config.

If you still need to wire `eval.py`/`infer.py`, follow `pipeline/REFACTOR_STATUS.md` — the
pattern is: add `--project DIR`; read prompt/model from `project.deploy`; read eval prompts from
`<project>/eval/prompts.jsonl`.

## Working agreements

- Don't add abstractions beyond what a stage needs. The schema is intentionally flat and
  data-class-y so a GUI can read/write it; keep it that way.
- The slur-strip / no-post-cutoff-figures / no-AI-acknowledgment requirements are content
  invariants for the *deployed* model and the *training data* — they belong in the project's
  `content_rules` (synthesis), `triage` rubric + critical flags, and the deploy-time moderation
  layer. Don't bake project-specific content policy into the `pipeline/` engine.
- `scratch/` is for working data and per-project material — never commit it.
