# voicepipe — GUI backend API & event schema

Status: **built.** `pipeline/events.py` (the event emitter) is wired into all stages;
`pipeline/jobs.py` (process supervision), `pipeline/server.py` (the FastAPI control server),
`pipeline/webui/` (the web UI), `pipeline/scaffold.py` + `pipeline/templates/` (project templates),
and a `desktop/` Tauri scaffold + `deploy/` (the web-only systemd packaging) all exist. Run it:
`voicepipe serve`. A few details differ from the sketch below — see **"Implementation notes"** at
the end; the rest of this doc is accurate.

## The shape of the thing

```
┌─────────────────────────────┐       HTTP + SSE        ┌────────────────────────────────┐
│  Desktop shell (Tauri/Wails)│  ───────────────────▶   │  pipeline/server.py (FastAPI)  │
│  - webview UI               │  ◀───────────────────   │  - REST for projects/jobs      │
│  - file dialogs             │                         │  - SSE stream of stage events  │
│  - "is the engine alive?"   │                         │  - job/process supervision     │
└─────────────────────────────┘                         └───────────────┬────────────────┘
                                                                        │ spawns
                                                          python -m pipeline <stage> --project DIR
                                                                        │  (NDJSON events on a pipe)
                                                              ┌─────────▼─────────┐
                                                              │  the 6 stages     │
                                                              │  (unchanged CLI)  │
                                                              └───────────────────┘
```

- **The engine stays 100% Python and 100% the same CLI.** A stage run is `python -m pipeline <stage> --project DIR`.
- **The desktop-shell language (Rust/Go) does almost nothing** — spawn/supervise the server, host the
  webview, native dialogs. It is *not* in the data path. Choose Tauri or Wails on developer preference;
  Tauri's sidecar mechanism is the better fit for bundling/supervising the Python side.
- **The GUI never imports Python.** It talks to the local FastAPI server over `127.0.0.1:<port>`. The
  server is also runnable headless and on a *remote* box — important here, because training runs on a
  different machine than the laptop.

## Why a local HTTP server (not just subprocess + stdout)

Training is a multi-hour job; the user closes the laptop lid; the GUI restarts. A server that owns the
job (PID, log buffer, event history) lets the GUI reconnect and replay. It also makes "drive a job on
the 5070 / a rented GPU" a deployment choice, not a rewrite. The price is one small new module
(`pipeline/server.py`, ~200 lines: FastAPI + a job table + the subprocess plumbing) plus a job-state
file under each project's `dataset/jobs/`.

---

## REST surface

Base: `http://127.0.0.1:<port>/v1`. JSON in/out. All paths project-scoped where it makes sense.

### Projects

| Method & path | Body / params | Returns |
|---|---|---|
| `GET /projects` | — | `[{id, name, path, description, last_modified}]` — `id` is a slug of the dir; `path` is absolute. Discovered by scanning a configurable set of roots (`~/voicepipe-projects`, `projects/`, `scratch/`). |
| `POST /projects` | `{path}` (existing dir) or `{name, template?}` (scaffold a new one) | `{id, ...}`. Scaffold writes a minimal `project.toml`, empty `corpus/`, `seeds/`, `prompts/`. |
| `GET /projects/{id}` | — | `{id, name, path, description, config, corpus_files, seed_count, dataset_state}`. `config` is the full resolved `Project` as JSON (text-reference fields resolved to their contents); `dataset_state` reports what artifacts exist (`raw: {batches, pairs}`, `dedup: {pairs}`, `triage: {scored, kept}`, `final: {train, val}`, `adapter: {has_gguf, has_modelfile}`). |
| `PUT /projects/{id}/config` | the `Project` JSON (or a partial patch) | `{config}` — writes `project.toml` back. Long prose blocks (`synth_preamble`, mode descriptions, …) can be sent inline; the server decides whether to inline them in the TOML or spill them to `prompts/<field>.md` and reference by filename (mirrors how a human writes it). Validates by round-tripping through `load_project`. |
| `GET /projects/{id}/files/{path}` | — | raw file contents (for the prose-block / corpus editors). |
| `PUT /projects/{id}/files/{path}` | raw contents | writes it. |
| `DELETE /projects/{id}` | `?delete_dataset=false` | unregister; optionally `rm -rf dataset/`. Never deletes `corpus/`/`seeds/`/`prompts/`. |

### Stage runs (jobs)

| Method & path | Body | Returns |
|---|---|---|
| `POST /projects/{id}/stages/{stage}/run` | `{overrides?: {...}, smoke?: bool, gpu?: int}` — `stage` ∈ `categorize\|synthesize\|dedup\|triage\|assemble\|train\|deploy\|eval`. `overrides` maps onto the stage's CLI flags (`{"max_seq_len": 4096}` → `--max-seq-len 4096`; `{"no_downsample": true}` → `--no-downsample`). | `201` `{job_id, stage, project, dir, command, pid, started_at, status:"running"}`. 409 if that stage is already running for this project. |
| `GET /jobs` | `?project=&status=&limit=` | `[{job_id, project, stage, status, started_at, ended_at, exit_code}]`. |
| `GET /jobs/{job_id}` | — | full record incl. the resolved command line, the last N log lines, the latest progress event, and the summary (once finished). |
| `GET /jobs/{job_id}/events` | `?since=<seq>` | **SSE stream.** Replays every event with `seq > since` from the persisted log, then live-tails. `seq` is a monotonic per-job integer. Each SSE `data:` line is one event object (schema below). Closes when the job ends (after emitting `stage_end`). |
| `GET /jobs/{job_id}/log` | `?tail=&from=` | the raw captured stdout+stderr (the human text the CLI prints), for a "show me the console" pane. |
| `POST /jobs/{job_id}:cancel` | — | SIGTERM the process group, then SIGKILL after a grace period. Emits a `stage_end` with `status:"cancelled"`. |

### Misc

| Method & path | Returns |
|---|---|
| `GET /health` | `{ok, version, python, has_torch, has_ollama, llama_cpp_dir}` — what's installed where (the train/deploy extras may be absent on a laptop). |
| `GET /providers` | the configured LLM providers (synthesis/triage models): `[{id, kind:"ollama_cloud"\|"openai"\|"anthropic"\|"local", base_url, models}]`. (Provider abstraction generalization is its own task; today this returns just the Ollama Cloud entry.) |
| `GET /defaults` | the dataclass-default `Project` as JSON — the GUI uses it to render "unset → (default: X)" hints in the config editor. |

---

## Event schema

Events are newline-delimited JSON. The stage process writes them to a sink chosen by env var (so the
CLI is unaffected when nothing sets it):

- `VOICEPIPE_EVENTS_FD=<n>` → write to that already-open file descriptor (the server passes a pipe). Preferred.
- else `VOICEPIPE_EVENTS_FILE=<path>` → append (for debugging / detached runs).
- else → no-op. (`VOICEPIPE_EVENTS_STDERR=1` additionally mirrors them to stderr.)

The server reads that pipe, assigns each event a per-job `seq`, persists it to `dataset/jobs/<job_id>/events.ndjson`, and fans it out to SSE subscribers.

### Common envelope

Every event has:

```jsonc
{
  "ts": "2026-05-12T07:14:22.118Z",   // ISO-8601 UTC, ms precision
  "stage": "synthesize",              // which stage emitted it
  "type": "progress",                 // see the type table
  // ...type-specific fields...
}
```

(`seq` and `job_id` are added by the server, not the stage.)

### Event types

| `type` | Emitted when | Key fields |
|---|---|---|
| `stage_start` | first thing a stage `main()` does | `command` (argv list), `params` (the effective resolved config for this run — e.g. `{model, target, concurrency}` for synth; `{base_model, lora_r, epochs, lr, ...}` for train), `inputs` (paths it will read), `outputs` (paths it will write), `resumed_from` (e.g. existing batch count) |
| `phase` | a stage moves between internal phases | `name` (`"hash_dedup"`, `"cosine_dedup"`, `"downsample"`, `"convert_gguf"`, `"ollama_create"`, `"ollama_push"`, …), `index`/`total` if phases are countable |
| `progress` | incrementally, throughout the long loop | `current`, `total` (may be `null` if unknown), `unit` (`"pairs"`, `"batches"`, `"rows"`, `"steps"`, `"bytes"`), `rate` (per-sec, optional), `eta_sec` (optional), `detail` (free string, e.g. `"mode=BIOGRAPHICAL parsed 11/12"`) |
| `metric` | training: on each `on_log` from the HF trainer | `step`, `epoch` (float), `loss`, `learning_rate`, `grad_norm`, plus eval metrics on eval steps (`eval_loss`, …). The GUI plots these. |
| `artifact` | a file/dir the run produced (so the GUI can offer "open"/"reveal") | `path` (absolute), `kind` (`"dataset"`, `"adapter"`, `"gguf"`, `"modelfile"`, `"stats"`, `"raw_dump"`), `bytes` (optional) |
| `log` | for the structured-ish notices worth surfacing distinctly from raw stdout | `level` (`"debug"\|"info"\|"warn"\|"error"`), `message` |
| `stage_end` | last thing, in a `finally` | `status` (`"ok"\|"error"\|"cancelled"`), `exit_code`, `duration_sec`, `summary` (a dict — stage-specific: synth → `{pairs_total, batches, by_mode}`; dedup → `{in, after_hash, after_cosine, after_downsample}`; triage → `{scored, kept, dropped_low, dropped_flagged}`; assemble → the `stats.json` contents; train → `{steps, final_loss, mean_loss, adapter_dir}`; deploy → `{tag, gguf, pushed}`), `error` (string + optional traceback when `status=="error"`) |

### Per-stage progress semantics (what `current/total/unit` mean)

| stage | `unit` | `total` | notes |
|---|---|---|---|
| `categorize` | `"corpus_files"` then `"categories"` | known | (stub today) |
| `synthesize` | `"pairs"` | `synthesis.target` | one `progress` per completed batch; `detail` carries the per-batch parse count; `resumed_from` on `stage_start` if `dataset/raw` already had pairs |
| `dedup` | `"pairs"` | the running count | a `phase` event before each of hash / cosine / downsample, with the surviving count after each as `progress` |
| `triage` | `"batches"` | `ceil(n_pairs / batch_size)` | one `progress` per completed batch; `detail` = `"parsed 28/30"`; resume-aware (already-scored batches counted at start) |
| `assemble` | `"rows"` | total assembled | mostly a single quick burst; `summary` carries the full stats |
| `train` | `"steps"` | `num_train_epochs * ceil(n_train / (batch * grad_accum))` (or `max_steps` in smoke) | `progress` on each logging step; `metric` alongside it; one `phase` for `"load_base_model"`, `"build_peft"`, `"train"`, `"save"` |
| `deploy` | n/a (phase-driven) | — | `phase` for `convert_gguf` → `ollama_create` → (`ollama_push`); `convert_gguf` may emit byte-progress if cheap to get; `artifact` for the gguf, the Modelfile, and the created tag |
| `eval` | `"prompts"` then `"comparisons"` | known | (not `--project`-wired yet) |

### Example: a synthesize run, abridged

```jsonc
{"ts":"…","stage":"synthesize","type":"stage_start","command":["python","-m","pipeline.synthesize","--project","scratch/dec-bot"],"params":{"model":"mistral-large-3:675b-cloud","target":3000,"concurrency":3,"balance_modes":true},"inputs":["scratch/dec-bot/corpus","scratch/dec-bot/seeds/seed_pairs_no_safety.jsonl"],"outputs":["scratch/dec-bot/dataset/raw"],"resumed_from":{"pairs_on_disk":1200,"next_batch_id":120}}
{"ts":"…","stage":"synthesize","type":"progress","current":1211,"total":3000,"unit":"pairs","detail":"batch 120 mode=COSMOLOGY parsed 11/12"}
{"ts":"…","stage":"synthesize","type":"artifact","path":"/…/dataset/raw/batch_00120.jsonl","kind":"dataset"}
{"ts":"…","stage":"synthesize","type":"log","level":"warn","message":"batch 123 under-parsed (3/12) — raw dump saved"}
{"ts":"…","stage":"synthesize","type":"artifact","path":"/…/dataset/raw/_raw_debug/batch_00123.txt","kind":"raw_dump"}
{"ts":"…","stage":"synthesize","type":"progress","current":3002,"total":3000,"unit":"pairs","detail":"batch 178 mode=PERSECUTION parsed 12/12"}
{"ts":"…","stage":"synthesize","type":"stage_end","status":"ok","exit_code":0,"duration_sec":4123.6,"summary":{"pairs_total":3002,"batches":59,"by_mode":{"COSMOLOGY":760,"PERSECUTION":748,"HISTORICAL_INDICTMENT":751,"BIOGRAPHICAL":743}}}
```

### Example: a train run, abridged

```jsonc
{"ts":"…","stage":"train","type":"stage_start","params":{"base_model":"unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit","lora_r":128,"lora_alpha":256,"max_seq_len":4096,"batch_size":8,"grad_accum":2,"epochs":4.0,"lr":2e-4,"optim":"paged_adamw_8bit"},"inputs":["…/dataset/final/train.jsonl","…/dataset/final/val.jsonl"],"outputs":["…/dataset/adapter"]}
{"ts":"…","stage":"train","type":"phase","name":"load_base_model"}
{"ts":"…","stage":"train","type":"phase","name":"build_peft"}
{"ts":"…","stage":"train","type":"log","level":"info","message":"trainable params: 228.4M / 12.18B (1.88%)"}
{"ts":"…","stage":"train","type":"phase","name":"train"}
{"ts":"…","stage":"train","type":"metric","step":5,"epoch":0.02,"loss":1.83,"learning_rate":1.6e-5,"grad_norm":0.9}
{"ts":"…","stage":"train","type":"progress","current":5,"total":424,"unit":"steps"}
…
{"ts":"…","stage":"train","type":"metric","step":105,"epoch":1.0,"eval_loss":0.71}
…
{"ts":"…","stage":"train","type":"phase","name":"save"}
{"ts":"…","stage":"train","type":"artifact","path":"…/dataset/adapter/final","kind":"adapter"}
{"ts":"…","stage":"train","type":"stage_end","status":"ok","exit_code":0,"duration_sec":1487.0,"summary":{"steps":424,"final_loss":0.34,"mean_loss":0.69,"adapter_dir":"…/dataset/adapter/final"}}
```

---

## Implementation order (the keystone-first plan)

1. **`pipeline/events.py` + wire the 6 stages** — the emitter (env-var sink, thread-safe, no-op when unset) and `emit(...)` calls at the structurally-important points (`stage_start` / `phase` / `progress` / `artifact` / `log` / `stage_end`) in `synthesize`, `dedup`, `triage`, `assemble`, `train` (incl. a `TrainerCallback` for `metric`), `deploy`. Existing `print()`s stay (they're the `log`-stream's raw form, captured from stdout by the server). ← **doing this now**
2. **A job/process layer** — `pipeline/jobs.py`: spawn a stage as a process group with `VOICEPIPE_EVENTS_FD` pointed at a pipe, capture stdout/stderr, persist `dataset/jobs/<id>/{meta.json,events.ndjson,console.log}`, support cancel, survive a server restart (re-attach by PID or mark crashed).
3. **`pipeline/server.py`** — FastAPI over (1)+(2): the REST surface above + the SSE endpoint. `voicepipe serve --port N` (add to `cli.py`).
4. **`categorize.py`** — implement for real (propose categories from a corpus); it's the zero-config promise.
5. **Generalize providers** — beyond Ollama Cloud.
6. **`eval.py` / `infer.py`** — `--project` wiring.
7. **The Tauri/Wails app** — project list, config editor (forms from the dataclass schema + `/defaults`), run/monitor screens consuming the SSE stream, console pane, loss-curve charts.
8. **Packaging** — ship the lightweight engine; train/deploy extras pip-installed on first use or run on a remote engine.

---

## Implementation notes (what shipped, vs. the sketch above)

- **Stage-run path** is `POST /v1/projects/{id}/stages/{stage}/run` (not `…:run` — avoids the
  colon-in-path routing wrinkle). Body: `{overrides?, smoke?, gpu?}`. Returns `201` with the job
  meta. `eval`/`infer` runs work too (eval isn't `--project`-wired yet, so it'll mostly no-op).
- **Auth** is a single shared token (`--auth-token` / `$VOICEPIPE_AUTH_TOKEN`). It's enforced on
  `/v1/*` and **only over TCP** — required automatically for any non-loopback `--host`, optional
  for `127.0.0.1`, and **bypassed entirely over a Unix socket** (`voicepipe serve --unix-socket
  PATH`). EventSource can't set headers, so the SSE endpoint also accepts `?token=`. Static UI is
  always served (so the login prompt can load); the UI stores the token in `localStorage`.
- **Config writes** (`PUT /v1/projects/{id}/config`): long prose (`synth_preamble`,
  `variety_menus`, `content_rules`, each mode's `description`, `triage.rubric`,
  `deploy.system_message`) is spilled to `prompts/*.md` and referenced by filename; everything
  else is written inline. Caveat: the TOML re-serializer is minimal — comments and literal-string
  quoting in a hand-authored `project.toml` are **not** preserved after a GUI save. GUI-created
  projects are fine; for hand-tuned ones (like `scratch/dec-bot/`), prefer editing the file.
- **Jobs** live under `<project>/dataset/jobs/<job_id>/` (`meta.json` / `events.ndjson` /
  `console.log`). `seq` is the 1-based line number in `events.ndjson`. The server tails the file
  for the SSE stream; on restart it re-scans job dirs and marks dead `running` jobs `crashed`.
- **`categorize`** is implemented (LLM proposes weighted prompt categories from the corpus →
  `dataset/categorize/proposed_categories.json`; `--adopt` writes them into `project.toml`).
- **Providers**: there's a `pipeline/providers/` abstraction (`get_chat_provider(spec)` →
  Ollama Cloud or any OpenAI-compatible endpoint incl. local Ollama / vLLM / LM Studio). The
  synthesis/triage stages still call the Ollama Cloud client directly for now; `/v1/providers`
  reports what's configured. Generalizing the stages to honor a per-project provider is a small
  follow-up.
- **`eval.py`** is still not `--project`-wired (the only stage that isn't). `infer.py` is.

## Two packaging artifacts (see `deploy/PACKAGING.md`)

1. **Native desktop app** (`desktop/`, Tauri): a webview + a `voicepipe serve` sidecar on an
   ephemeral **loopback** port — **no auth**, **no systemd**, killed on app exit. Ships a bundled
   standalone Python so the user installs nothing; does *not* bundle the heavy `[train]`/`[deploy]`
   extras — for fine-tuning it offers "connect to a remote engine" (point at a web-deployment URL).
   Refinement TODO: bind a Unix socket and bridge it from the Rust side so there's no TCP port at all.
2. **Web-only** (`deploy/`): `pip install -e .[gui]` + `deploy/voicepipe.service`, bound to a
   `--host:--port` with a `--auth-token`. **The only artifact with a systemd service.** No GUI app,
   no bundled Python — "the engine, reachable from a browser." Put it behind a TLS reverse proxy if
   exposed beyond a trusted LAN. `deploy/install-web.sh` sets it up (venv, install, generated token,
   unit).

The auth pipeline runs **only on the web path**; the native app uses the loopback/UDS path and
never prompts.
