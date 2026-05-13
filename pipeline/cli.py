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
"""

import sys


# pipeline stages: command -> module with a main()
_STAGES = {
    "synthesize": "pipeline.synthesize",
    "dedup": "pipeline.dedup",
    "triage": "pipeline.triage",
    "assemble": "pipeline.assemble",
    "train": "pipeline.train",
    "deploy": "pipeline.deploy",
    "eval": "pipeline.eval",
    "infer": "pipeline.infer",
    "categorize": "pipeline.categorize",
}
# project-lifecycle commands handled inline
_COMMANDS = set(_STAGES) | {"new", "serve"}


def _cmd_new(argv):
    import argparse
    from pathlib import Path
    from pipeline import scaffold
    ap = argparse.ArgumentParser(prog="voicepipe new", description="scaffold a new voicepipe project")
    ap.add_argument("name", help="project name (also used for the directory)")
    ap.add_argument("--template", "-t", default="character", help=f"template ({', '.join(t['id'] for t in scaffold.list_templates())})")
    ap.add_argument("--description", "-d", default="", help="one-line description")
    ap.add_argument("--dir", default=None, help="destination directory (default: ./<slug-of-name>)")
    ap.add_argument("--list-templates", action="store_true", help="list templates and exit")
    args = ap.parse_args(argv)
    if args.list_templates:
        for t in scaffold.list_templates():
            print(f"  {t['id']:<12} {t['description']}")
        return 0
    import re
    slug = re.sub(r"[^a-z0-9]+", "-", args.name.lower()).strip("-") or "project"
    dest = Path(args.dir) if args.dir else Path.cwd() / slug
    try:
        scaffold.create_project(dest, template=args.template, name=args.name, description=args.description)
    except (FileExistsError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"created project at {dest}")
    print("next:")
    print(f"  - put source texts in {dest}/corpus/  and a few example pairs in {dest}/seeds/seed_pairs.jsonl")
    print(f"  - edit {dest}/project.toml  (or run `voicepipe serve` and use the GUI)")
    print(f"  - then: voicepipe synthesize --project {dest}  ->  dedup  ->  triage  ->  assemble  ->  train  ->  deploy")
    return 0


def _cmd_serve(argv):
    from pipeline.server import main as serve_main
    return serve_main(argv) or 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd = argv[0]
    if cmd == "new":
        return _cmd_new(argv[1:])
    if cmd == "serve":
        return _cmd_serve(argv[1:])
    if cmd not in _STAGES:
        print(f"unknown command {cmd!r}.\ncommands: {', '.join(sorted(_COMMANDS))}", file=sys.stderr)
        return 2
    try:
        from pipeline.util import load_env_file
        load_env_file()        # pick up OLLAMA_API_KEY etc. from ~/.config/voicepipe/env
    except Exception:
        pass
    import importlib
    mod = importlib.import_module(_STAGES[cmd])
    sys.argv = [f"voicepipe {cmd}"] + argv[1:]
    return mod.main() or 0


if __name__ == "__main__":
    sys.exit(main())
