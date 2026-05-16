"""
voicepipe CLI: `voicepipe <command> [args...]`  (also `python -m pipeline <command> ...`)

Project lifecycle:
  new          scaffold a new project directory from a template      `voicepipe new my-character`
  serve        run the local control server + web GUI                 `voicepipe serve`

The corpus -> character pipeline (each takes `--project DIR`, in order):
  categorize   propose prompt categories from the corpus (LLM)        [scaffold — see categorize.py]
  synthesize   generate (user, response) pairs in the target voice
  dedup        cosine-similarity dedup + over-saturated-phrase caps
  triage       LLM-judge score 1-5 + policy flags; keep the good pairs
  assemble     combine kept synth + seeds + salvage -> train/val split
  train        QLoRA fine-tune a base model on the assembled dataset  (needs the [train] extra + a CUDA box)
  deploy       LoRA -> GGUF -> Modelfile -> ollama create -> (push)   (needs the [deploy] extra + a llama.cpp checkout)
  eval         LLM-judge eval (absolute + pairwise)
  infer        quick inference test grid

Run `voicepipe <command> --help` for command-specific flags.
The GUI (`voicepipe serve`) drives exactly these same stages — nothing the GUI can do is unavailable here.

Implementation note: the command layer is built on `clir` (ClirApp). Each command below maps its
typed clir params into an argparse.Namespace and hands it to the matching stage module's
`main(args)`. The stage modules still accept `main()` with no args, so `python -m pipeline.<stage>`
keeps working as a standalone entry point.
"""

import sys
from argparse import Namespace

from clir import ClirApp, argument, option
from clir.errors import ClirError


def _load_env():
    """Pick up OLLAMA_API_KEY etc. from ~/.config/voicepipe/env (best-effort)."""
    try:
        from pipeline.util import load_env_file
        load_env_file()
    except Exception:
        pass


def _version():
    try:
        from pipeline import __version__
        return __version__
    except Exception:
        return "0+unknown"


def _run_stage(module_name, args):
    """Import a stage module, run its main(args), translate a nonzero return into an exit code."""
    _load_env()
    import importlib
    mod = importlib.import_module(module_name)
    rc = mod.main(args) or 0
    if rc:
        raise ClirError(f"{module_name.rsplit('.', 1)[-1]} exited with status {rc}", exit_code=rc)
    return None


app = ClirApp(
    name="voicepipe",
    description=__doc__,
    version=_version(),
)


# ── pipeline stages ───────────────────────────────────────────────────────────

@app.command(help="generate (user, response) pairs in the target voice")
@option("--project", required=True, help="project directory (reads project.toml)")
@option("--target", type=int, help="override total-pairs target")
@option("--pairs-per-batch", type=int)
@option("--model")
@option("--concurrency", type=int)
@option("--mode", help="force every batch to this mode")
@option("--no-balance-modes", type=bool, default=False,
        help="weighted-random mode selection instead of round-robin")
@option("--temperature", type=float)
def synthesize(project, target, pairs_per_batch, model, concurrency, mode,
               no_balance_modes, temperature):
    """generate (user, response) pairs in the target voice"""
    _run_stage("pipeline.synthesize", Namespace(
        project=project, target=target, pairs_per_batch=pairs_per_batch,
        model=model, concurrency=concurrency, mode=mode,
        no_balance_modes=no_balance_modes, temperature=temperature,
    ))


@app.command(help="cosine-similarity dedup + over-saturated-phrase caps")
@option("--project", help="project directory (reads project.toml → DedupConfig)")
@option("--raw-dir", help="override input dir of batch_*.jsonl")
@option("--out", help="override output pairs.jsonl path")
@option("--threshold", type=float, help="override cosine threshold")
@option("--skip-embed", type=bool, default=False, help="hash-only dedup")
@option("--no-downsample", type=bool, default=False, help="skip closer caps")
def dedup(project, raw_dir, out, threshold, skip_embed, no_downsample):
    """cosine-similarity dedup + over-saturated-phrase caps"""
    _run_stage("pipeline.dedup", Namespace(
        project=project, raw_dir=raw_dir, out=out, threshold=threshold,
        skip_embed=skip_embed, no_downsample=no_downsample,
    ))


@app.command(help="LLM-judge score 1-5 + policy flags; keep the good pairs")
@option("--project", help="project directory (reads project.toml → TriageConfig)")
@option("--in", dest="in_path", help="override input dedup pairs.jsonl")
@option("--out-dir", help="override output triage dir")
@option("--model")
@option("--batch-size", type=int)
@option("--min-keep", type=int)
@option("--concurrency", type=int)
@option("--max-pairs", type=int, help="dry-run cap")
def triage(project, in_path, out_dir, model, batch_size, min_keep,
           concurrency, max_pairs):
    """LLM-judge score 1-5 + policy flags; keep the good pairs"""
    _run_stage("pipeline.triage", Namespace(
        project=project, in_path=in_path, out_dir=out_dir, model=model,
        batch_size=batch_size, min_keep=min_keep, concurrency=concurrency,
        max_pairs=max_pairs,
    ))


@app.command(help="combine kept synth + seeds + salvage -> train/val split")
@option("--project", help="project directory (reads project.toml → AssembleConfig)")
@option("--keep", help="override path to triage keep.jsonl")
@option("--seeds", help="override path to seeds jsonl")
@option("--salvage", multiple=True, help="override salvage jsonl path(s); repeatable")
@option("--out-dir", help="override output dir")
@option("--val-fraction", type=float,
        help="fraction of pairs reserved for val.jsonl (honored strictly; a stderr "
             "warning is emitted if the resulting val set is < 20 pairs)")
@option("--seed", type=int)
def assemble(project, keep, seeds, salvage, out_dir, val_fraction, seed):
    """combine kept synth + seeds + salvage -> train/val split"""
    _run_stage("pipeline.assemble", Namespace(
        project=project, keep=keep, seeds=seeds,
        salvage=(list(salvage) if salvage else None),
        out_dir=out_dir, val_fraction=val_fraction, seed=seed,
    ))


@app.command(help="QLoRA fine-tune a base model on the assembled dataset")
@option("--gpu", type=int,
        help="pin to this physical GPU (sets CUDA_VISIBLE_DEVICES); for running two trainings on a 2-GPU box")
@option("--project", help="project directory (reads project.toml → TrainConfig + dataset paths)")
@option("--model", help="override: pre-quantized 4-bit base model on HF Hub")
@option("--train-jsonl", help="override path to train.jsonl")
@option("--val-jsonl", help="override path to val.jsonl")
@option("--out-dir", help="override adapter output directory")
@option("--epochs", type=float)
@option("--batch-size", type=int)
@option("--grad-accum", type=int)
@option("--lr", type=float)
@option("--lora-r", type=int)
@option("--lora-alpha", type=int)
@option("--lora-dropout", type=float)
@option("--max-seq-len", type=int)
@option("--optim")
@option("--resume-adapter",
        help="continue training an existing LoRA adapter dir (weights only — discards optimizer/scheduler/RNG state)")
@option("--resume-from-checkpoint",
        help="resume optimizer+scheduler+RNG state from a Trainer checkpoint dir; "
             "pass 'auto' to pick the latest out_dir/checkpoint-* automatically")
@option("--smoke", type=bool, default=False,
        help="10-step dry run: tiny subset, verifies load + GPU + pipeline")
def train(gpu, project, model, train_jsonl, val_jsonl, out_dir, epochs,
          batch_size, grad_accum, lr, lora_r, lora_alpha, lora_dropout,
          max_seq_len, optim, resume_adapter, resume_from_checkpoint, smoke):
    """QLoRA fine-tune a base model on the assembled dataset"""
    _run_stage("pipeline.train", Namespace(
        gpu=gpu, project=project, model=model, train_jsonl=train_jsonl,
        val_jsonl=val_jsonl, out_dir=out_dir, epochs=epochs,
        batch_size=batch_size, grad_accum=grad_accum, lr=lr, lora_r=lora_r,
        lora_alpha=lora_alpha, lora_dropout=lora_dropout, max_seq_len=max_seq_len,
        optim=optim, resume_adapter=resume_adapter,
        resume_from_checkpoint=resume_from_checkpoint, smoke=smoke,
    ))


@app.command(help="LoRA -> GGUF -> Modelfile -> ollama create -> (push)")
@option("--project", required=True, help="project directory (reads project.toml → DeployConfig)")
@option("--adapter",
        help="HF PEFT LoRA adapter dir (default: <project>/dataset/adapter/final, where `train` writes it)")
@option("--llama-cpp-dir", help="path to a llama.cpp checkout (has convert_lora_to_gguf.py)")
@option("--outdir", help="where to write the GGUF + Modelfile (default: <project>/dataset/adapter)")
@option("--tag", help="override the Ollama tag")
@option("--push", type=bool, default=False, help="`ollama push` the created tag afterward")
@option("--dry-run", type=bool, default=False,
        help="write the GGUF + Modelfile but don't `ollama create`")
def deploy(project, adapter, llama_cpp_dir, outdir, tag, push, dry_run):
    """LoRA -> GGUF -> Modelfile -> ollama create -> (push)"""
    _run_stage("pipeline.deploy", Namespace(
        project=project, adapter=adapter, llama_cpp_dir=llama_cpp_dir,
        outdir=outdir, tag=tag, push=push, dry_run=dry_run,
    ))


@app.command(help="LLM-judge eval (absolute + pairwise)")
@option("--project",
        help="project dir: defaults --model to its deploy.ollama_tag, --prompts to "
             "<project>/eval/prompts.jsonl, output to <project>/dataset/eval/, and loads "
             "eval/judge_rubric.md / eval/judge_pairwise.md if present (else falls back to "
             "built-in dec-bot defaults)")
@option("--model", help="model name (absolute mode); resolved via the configured provider")
@option("--compare", nargs=2, help="Compare two models A and B (pairwise mode)")
@option("--judge", default="minimax-m2.7:cloud", help="model name used as judge")
@option("--prompts", help="JSONL with eval prompts (default: <project>/eval/prompts.jsonl)")
@option("--system", help="Override the candidate's system prompt (default: model default)")
@option("--temperature", type=float, default=0.7)
@option("--max-tokens", type=int, default=500)
@option("--concurrency", type=int, default=3,
        help="Parallel in-flight calls. 3 for cloud thinking-model judge; higher for local-only.")
@option("--seed", type=int, default=42, help="Random seed for pairwise A/B flip ordering")
@option("--tag", help="Tag for the output filename (default: model name or A-vs-B)")
def eval(project, model, compare, judge, prompts, system, temperature,
         max_tokens, concurrency, seed, tag):
    """LLM-judge eval (absolute + pairwise)"""
    from pathlib import Path
    _run_stage("pipeline.eval", Namespace(
        project=project, model=model,
        compare=(list(compare) if compare else None),
        judge=judge,
        prompts=(Path(prompts) if prompts else None),
        system=system, temperature=temperature, max_tokens=max_tokens,
        concurrency=concurrency, seed=seed, tag=tag,
    ))


@app.command(help="quick inference test grid")
@option("--project", help="project dir: take base model / system prompt / eval prompts from it")
@option("--base", help="override base model (pre-quantized 4-bit HF id)")
@option("--adapter",
        help="LoRA adapter dir (default: <project>/dataset/adapter/final, or ./checkpoints/final)")
@option("--system", help="override the system prompt (or '' for none)")
@option("--max-new-tokens", type=int, default=320)
def infer(project, base, adapter, system, max_new_tokens):
    """quick inference test grid"""
    _run_stage("pipeline.infer", Namespace(
        project=project, base=base, adapter=adapter, system=system,
        max_new_tokens=max_new_tokens,
    ))


@app.command(help="propose prompt categories from the corpus (LLM)")
@option("--project", required=True, help="project directory")
@option("--n", type=int, default=12, help="how many categories to propose")
@option("--model", help="override the proposing model (default: project synthesis.model)")
@option("--adopt", type=bool, default=False,
        help="write the proposal into project.toml's `categories`")
def categorize(project, n, model, adopt):
    """propose prompt categories from the corpus (LLM)"""
    _run_stage("pipeline.categorize", Namespace(
        project=project, n=n, model=model, adopt=adopt,
    ))


# ── project lifecycle ─────────────────────────────────────────────────────────

@app.command(help="scaffold a new voicepipe project directory from a template")
@argument("name", required=True, help="project name (also used for the directory)")
@option("--template", "-t", default="character", help="template (character, blank)")
@option("--description", "-d", default="", help="one-line description")
@option("--dir", help="destination directory (default: ./<slug-of-name>)")
@option("--list-templates", type=bool, default=False, help="list templates and exit")
def new(name, template, description, dir, list_templates):
    """scaffold a new voicepipe project directory from a template"""
    import re
    from pathlib import Path
    from pipeline import scaffold

    if list_templates:
        for t in scaffold.list_templates():
            print(f"  {t['id']:<12} {t['description']}")
        return

    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "project"
    dest = Path(dir) if dir else Path.cwd() / slug
    try:
        scaffold.create_project(dest, template=template, name=name, description=description)
    except (FileExistsError, ValueError) as e:
        raise ClirError(str(e), exit_code=2)
    print(f"created project at {dest}")
    print("next:")
    print(f"  - put source texts in {dest}/corpus/  and a few example pairs in {dest}/seeds/seed_pairs.jsonl")
    print(f"  - edit {dest}/project.toml  (or run `voicepipe serve` and use the GUI)")
    print(f"  - then: voicepipe synthesize --project {dest}  ->  dedup  ->  triage  ->  assemble  ->  train  ->  deploy")


@app.command(help="run the local control server + web GUI")
@option("--host", default="127.0.0.1",
        help="bind host (default 127.0.0.1; use 0.0.0.0 for remote)")
@option("--port", type=int, default=8765)
@option("--unix-socket",
        help="bind a Unix domain socket instead of host:port (no auth; for the native app)")
@option("--auth-token",
        help="require this token on every request (Authorization: Bearer ...). "
             "Default: $VOICEPIPE_AUTH_TOKEN")
@option("--no-auth", type=bool, default=False,
        help="disable auth entirely (loopback only; refuses non-loopback binds)")
@option("--root", multiple=True,
        help="project root to scan (repeatable). "
             "Default: cwd, ~/voicepipe-projects, ./projects, ./scratch")
@option("--no-browser", type=bool, default=False, help="don't try to open a browser")
def serve(host, port, unix_socket, auth_token, no_auth, root, no_browser):
    """run the local control server + web GUI"""
    import os
    from pipeline.server import main as serve_main
    if auth_token is None:
        auth_token = os.environ.get("VOICEPIPE_AUTH_TOKEN")
    rc = serve_main(Namespace(
        host=host, port=port, unix_socket=unix_socket, auth_token=auth_token,
        no_auth=no_auth, root=(list(root) if root else None), no_browser=no_browser,
    )) or 0
    if rc:
        raise ClirError(f"serve exited with status {rc}", exit_code=rc)


def main(argv=None):
    """Entry point for the `voicepipe` console script and `python -m pipeline`."""
    app.run(argv if argv is not None else sys.argv[1:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
