# Refactor status

## Config-driven + GUI-drivable (done)

The whole pipeline is `--project DIR`-driven (config from `project.toml`, schema in
`pipeline/project.py`), emits structured progress events, and is exposed by both the CLI and the
web GUI:

- `pipeline/project.py` — the `Project` schema + `load_project(dir)` + `glossary_text`/`load_seeds`/`load_corpus`
- `pipeline/synthesize.py` — modes/categories/styles/length-profiles/prompts/synthesis-config from the project; rolling-submission concurrency; the 4-strategy response parser; mode-conditional context dossiers
- `pipeline/dedup.py` — cosine threshold, closer patterns (name/regex/cap), embed model/url from `project.dedup`; phase events for hash/cosine/downsample
- `pipeline/triage.py` — rubric, model, critical flags, batch size, min_keep, concurrency from `project.triage`; resume-from-scored; finalize keep.jsonl
- `pipeline/assemble.py` — val_fraction, seed, salvage_paths, seeds_file from `project.assemble`
- `pipeline/train.py` — base_model + all LoRA/optimizer hyperparams + dataset paths from `project.train`; CausalLM / multimodal / unsloth loader fallbacks; `--resume-adapter`; a `TrainerCallback` emits `metric` events (loss/lr/grad_norm/eval_loss) for the GUI's loss chart
- `pipeline/deploy.py` — LoRA→GGUF → Modelfile → `ollama create` (+ `--push`); config from `project.deploy`; phase events for convert/create/push
- `pipeline/categorize.py` — **implemented**: an LLM proposes weighted prompt categories from the corpus → `dataset/categorize/proposed_categories.json`; `--adopt` writes them into `project.toml`
- `pipeline/events.py` — structured NDJSON event stream (stage_start/phase/progress/metric/artifact/log/stage_end); sink chosen by env var (`VOICEPIPE_EVENTS_FD` / `VOICEPIPE_EVENTS_FILE`); no-op for plain CLI use. Schema: `pipeline/GUI_API.md`
- `pipeline/jobs.py` — `JobManager`: run a stage as a tracked subprocess (detached process group), capture `meta.json`/`events.ndjson`/`console.log` under `<project>/dataset/jobs/<id>/`, cancel, survive a server restart
- `pipeline/server.py` — FastAPI control server + web-UI host (`voicepipe serve`); REST under `/v1/`, an SSE event stream, optional shared-token auth (TCP only; bypassed over a Unix socket)
- `pipeline/webui/` — the web UI (dependency-free HTML/CSS/JS): project list, a config editor generated from the schema, a pipeline view, a live job monitor with a loss chart
- `pipeline/scaffold.py` + `pipeline/templates/{character,blank}/` — `voicepipe new` / the GUI's "New project"
- `pipeline/providers/` — provider abstraction: `get_chat_provider(spec)` → Ollama Cloud or any OpenAI-compatible endpoint (`openai_compat.py`)
- `pipeline/cli.py` / `__main__.py` — `voicepipe <command>` (stages + `new` + `serve`)
- `pipeline/util.py` — shared helpers
- `scratch/dec-bot/project.toml` + `scratch/dec-bot/prompts/` — the worked example
- `desktop/` — Tauri (Rust) shell scaffold; `deploy/` — the web-only systemd packaging (unit + install script + `PACKAGING.md`)

A new character: `voicepipe new <name>` (or copy a project dir into `projects/` / `scratch/`),
add a corpus + seeds, then `voicepipe synthesize --project ...` → ... → `train` → `deploy` — or do
it all in `voicepipe serve`.

## Still loose / not done

- `pipeline/eval.py` — LLM-judge eval (absolute + pairwise) is **not yet `--project`-wired**
  (the only stage that isn't). Needs: `--project` → use `project.deploy.system_message` /
  `.ollama_tag`, read prompts from `<project>/eval/prompts.jsonl`. (`infer.py` *is* wired now.)
- The synthesis/triage stages still construct `OllamaCloudClient()` directly rather than going
  through `pipeline.providers.get_chat_provider(...)` with a per-project provider spec — small
  follow-up; the abstraction + an OpenAI-compatible impl already exist.
- `pipeline/server.py`'s TOML re-serializer is minimal — a GUI "save config" on a hand-authored
  `project.toml` drops comments and literal-string quoting. Fine for GUI-created projects; edit
  hand-tuned ones (like `scratch/dec-bot/`) as files.
- `pipeline/serve/discord_bot.py` is env-parametrized; its moderation patterns could become a
  `[deploy.moderation]` block.
- `desktop/` builds against a `voicepipe` on PATH; bundling a standalone Python and the
  "bind a Unix socket, no TCP port at all" refinement are TODO (see `desktop/README.md`).

## Project-specific material (in scratch/, .gitignored)

`scratch/dec-bot/` holds the dec-bot corpus/seeds/dataset, the worked `project.toml` + `prompts/`,
`scripts/salvage_v1.py`, `DEC_BOT_NOTES.md`, `WRITEUP.md`, and the deployed Modelfiles.
