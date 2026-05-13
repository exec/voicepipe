# voicepipe

A corpus-to-character pipeline. Point it at a corpus (a body of writing in a distinctive
voice), and it walks through: propose prompt categories → synthesize a fine-tuning dataset
in that voice → dedup it → triage it with an LLM judge → assemble train/val → QLoRA-train a
base model → deploy as an Ollama model. There's a CLI (`voicepipe <stage> --project DIR`) and
a local web GUI (`voicepipe serve`) — both drive the *same* engine. A fully-worked example
project ships in `examples/oscar-wilde/`; the original proof-of-concept (`scratch/dec-bot/`, not
in the repo) is a chatbot in the voice of Francis E. Dec, Esq.

## Models built with voicepipe

Both are public ollama tags — `ollama run <name>` to talk to either:

- [`execxd/mistral-nemo-12b-oscar-wilde`](https://ollama.com/execxd/mistral-nemo-12b-oscar-wilde)
  — Mistral Nemo 12B QLoRA fine-tune, in the voice of Oscar Wilde (1854–1900). End-to-end
  validation of the pipeline on a fresh character; the full project ships in
  [`examples/oscar-wilde/`](examples/oscar-wilde/).
- [`execxd/mistral-nemo-12b-francis-e-dec`](https://ollama.com/execxd/mistral-nemo-12b-francis-e-dec)
  — Mistral Nemo 12B QLoRA fine-tune, in the voice of Francis E. Dec, Esq. (1926–1996). The
  original proof-of-concept that drove the engine's design.

## Get it

**Desktop app** (recommended): grab the build for your OS (macOS `.dmg`, Windows `.msi`, Linux
`.AppImage`/`.deb`) from the releases page, open it, and configure + run the pipeline in the
window. It's self-contained — no Python or other setup needed. (macOS: it's not yet
notarized, so the first launch needs right-click → Open, or `xattr -dr com.apple.quarantine
/Applications/voicepipe.app`.) On first run, open **Settings** (bottom-left of the sidebar) and
paste your **Ollama Cloud API key** — synthesis, triage, and the category proposer need it.

**Or from source** (also gives you the CLI):

```bash
pip install -e ".[gui]"               # the engine + the web GUI/control server
voicepipe new my-character            # scaffold a project from a template (also: --template blank, --list-templates)
#   ...drop source texts in my-character/corpus/ , a few example pairs in my-character/seeds/seed_pairs.jsonl
export OLLAMA_API_KEY=ollama_...      # (or set it in the GUI's Settings)
voicepipe serve                       # http://127.0.0.1:8765 — configure + run the pipeline in the browser
# or on the CLI:
voicepipe synthesize --project my-character   # → dedup → triage → assemble → train → deploy
```

`voicepipe train` needs the CUDA-box extras: `pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0 && pip install -e ".[train]" -c constraints-train.txt`
(the constraints file pins the Blackwell-sm_120 known-good set). `voicepipe deploy` needs `[deploy]`
+ a llama.cpp checkout. Both can run on a **different machine** than the GUI — set its address in
the app's Settings → "Connect to a remote engine" (the box runs `voicepipe serve --host 0.0.0.0
--auth-token …`), and the GUI drives jobs there.

To build the desktop app yourself: `desktop/build-sidecar.sh && (cd desktop && cargo tauri build)`
— see `desktop/README.md`.

## Layout

```
pipeline/                  the generalized engine (a Python package)
  project.py               the Project config schema + load_project(dir)  ← the product's data model
  cli.py / __main__.py      `voicepipe <command>` / `python -m pipeline <command>`
  scaffold.py              create a project from a bundled template
  templates/               bundled project templates (character, blank)
  events.py                structured progress events (NDJSON; opt-in via env var; no-op for plain CLI)
  jobs.py                  process supervision — run a stage as a tracked subprocess
  server.py                FastAPI control server + the web GUI host  (`voicepipe serve`)
  webui/                   the web UI (dependency-free HTML/CSS/JS)
  synthesize.py            generate (user, response) pairs
  dedup.py                 cosine dedup + over-saturated-phrase caps
  triage.py                LLM-judge score 1-5 + policy flags
  assemble.py              combine kept synth + seeds + salvage → train/val
  train.py                 QLoRA fine-tune  ← config-driven, runs on a CUDA box
  deploy.py                LoRA → GGUF → Modelfile → ollama create → push  ← config-driven
  categorize.py            propose weighted prompt categories from a corpus (LLM)
  eval.py / infer.py       LLM-judge eval; quick inference test grid  (infer is --project-wired; eval not yet)
  providers/               LLM provider abstraction (Ollama Cloud; any OpenAI-compatible endpoint)
  serve/discord_bot.py     a Discord front-end (with an output moderation layer)
  GUI_API.md               the REST API + event schema (the GUI/CLI contract)
  REFACTOR_STATUS.md       what's config-driven, what isn't yet
desktop/                   Tauri (Rust) desktop shell — webview + a loopback `voicepipe serve` sidecar
deploy/                    the web-only deployment: a systemd unit + install script + PACKAGING.md
examples/                  shipped reference projects
  oscar-wilde/             a fully-worked configuration — corpus, seeds, prompts, project.toml
projects/                  per-project configs (your own go here)
scratch/                   .gitignored — working data, experiments, the dec-bot WIP
  dec-bot/
    project.toml           the dec-bot configuration (everything: modes, categories, hyperparams, ...)
    prompts/               long prose blocks referenced from project.toml
    corpus/ seeds/ dataset/  the dec-bot inputs and produced artifacts
    DEC_BOT_NOTES.md       project-specific guidance ; WRITEUP.md  the v1 runbook
pyproject.toml             package metadata; extras: [gui] [train] [serve] [deploy]
```

## A project

A project is a directory with a `project.toml` plus its corpus/seeds. `project.toml` carries
all the configuration — scalars and small lists inline, long prose blocks referenced by
filename (`prompts/*.md`). `pipeline.project.load_project(dir)` reads it and fills every
unspecified field with a default, so a minimal project is just `name`, a corpus, and a mode.
`voicepipe new <name>` scaffolds one; the GUI's "New project" does the same; or copy
`examples/oscar-wilde/` (the fully-worked example). See `pipeline/project.py` for the schema.

## Commands

```bash
voicepipe new NAME [--template character|blank] [--description ...]   # scaffold a project
voicepipe serve [--port 8765] [--host 0.0.0.0 --auth-token T] [--unix-socket PATH]   # control server + web GUI
voicepipe categorize  --project DIR [--n 12] [--adopt]               # propose prompt categories from the corpus
voicepipe synthesize  --project DIR                                  # generate (user, response) pairs
voicepipe dedup       --project DIR
voicepipe triage      --project DIR
voicepipe assemble    --project DIR
voicepipe train       --project DIR [--smoke] [--gpu N]              # QLoRA  (needs [train])
voicepipe deploy      --project DIR --adapter dataset/adapter/final --llama-cpp-dir PATH [--push]   # (needs [deploy])
voicepipe infer       --project DIR                                  # quick inference grid against the trained adapter
```

The GUI exposes exactly these — nothing it can do is unavailable on the CLI, and vice versa. Set
`VOICEPIPE_EVENTS_FILE=path` (or pass `--unix-socket`/run via the GUI) to get a structured NDJSON
progress stream out of any stage; without it the stages just print their usual human output.

## The GUI / control server

`voicepipe serve` runs a small FastAPI app: a REST API under `/v1/` (projects, config, stage
runs, jobs, an SSE event stream) plus the web UI at `/`. Bound to `127.0.0.1` it needs no auth;
bound to a public host it requires `--auth-token` (the web UI prompts for it). The desktop app
(`desktop/`) wraps the same server on a loopback port — see `deploy/PACKAGING.md` for the two
packaging stories (native app vs. web-only-with-systemd) and `pipeline/GUI_API.md` for the API.

## Status

- **Config-driven, end-to-end:** `project.py` (schema), `synthesize`, `dedup`, `triage`,
  `assemble`, `train`, `deploy`, `categorize`, `cli`, `events`, `jobs`, `server`, `scaffold`,
  `providers`. Each stage is `--project DIR`-driven and emits structured events. The
  `examples/oscar-wilde/` project is the worked example; two models have been built end-to-end
  with the pipeline (see "Models built with voicepipe" above).
- **Partial:** `eval.py` isn't `--project`-wired yet (`infer.py` is). `serve/discord_bot.py` is
  env-parametrized; its moderation patterns could become a project config block. The
  synthesis/triage stages call the Ollama Cloud client directly — generalizing them to honor a
  per-project provider entry is a small follow-up. The Tauri desktop scaffold builds against a
  PATH `voicepipe`; bundling a standalone Python (and the "bind a UDS, no TCP port" refinement)
  is TODO. See `pipeline/REFACTOR_STATUS.md` and `pipeline/GUI_API.md`.

## Content note

Every project ships absolute content rules in `prompts/content_rules.md` (enforced at synthesis
time) and the corresponding flags in `prompts/triage_rubric.md` (enforced at curation time): no
slurs, no group-attributed villainy, no post-cutoff references for the character's era, no
breaking frame, no serious real-world-harm instructions. These are the same rules the deployed
model is held to. The `character` template seeds them as defaults for any new project; the
`examples/oscar-wilde/` project carries them verbatim.

For voices that touch sensitive material — like the dec-bot proof-of-concept, whose source
corpus is paranoid-schizophrenic writing from the 1970s–80s — clean the corpus upstream and
add a deploy-time output moderation layer (see `serve/discord_bot.py` for the dec-bot's).
