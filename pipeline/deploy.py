"""
Deploy a trained LoRA adapter as an Ollama model.

Steps (config from project.toml [deploy], or CLI overrides):
  1. Convert the HF PEFT LoRA adapter -> GGUF via llama.cpp's convert_lora_to_gguf.py.
     bnb-quantized bases need --base-model-id pointed at a non-quantized HF mirror
     (DeployConfig.base_model_id_override) so the converter can read the base config.
  2. Write a Modelfile: FROM <ollama_from> + ADAPTER <gguf> + SYSTEM <deploy system prompt>
     + PARAMETER lines + PARAMETER stop "...".
  3. `ollama create <ollama_tag> -f <Modelfile>`  (and optionally `ollama push`).

Run on a box with `ollama` and a llama.cpp checkout (DeployConfig.llama_cpp_dir, or --llama-cpp-dir).

Usage:
  python -m pipeline deploy --project scratch/dec-bot --adapter <hf_adapter_dir> --llama-cpp-dir /path/to/llama.cpp [--push]
"""

import argparse
import json
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

from pipeline import events


def _slug(name: str) -> str:
    """Filename-safe slug for project names. `"Oscar Wilde"` -> `"oscar-wilde"`. Never empty."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", name or "").strip("-").lower()
    return s or "model"


def _modelfile_text(deploy_cfg, gguf_path: Path) -> str:
    """Render Ollama Modelfile text. Triple-quote-safe SYSTEM body and JSON-quoted stop strings."""
    lines = [f"FROM {deploy_cfg.ollama_from}", f"ADAPTER {gguf_path}", ""]
    if deploy_cfg.system_message:
        # `SYSTEM """..."""` parsing breaks if the body itself contains `"""`. Escape any inner
        # triple-quote runs by inserting a zero-width-ish backslash sequence; the rendered text
        # remains visually identical to the model at inference time (Ollama strips the escapes).
        safe = deploy_cfg.system_message.replace('"""', '\\"\\"\\"')
        lines += [f'SYSTEM """{safe}"""', ""]
    for k, v in (deploy_cfg.parameters or {}).items():
        lines.append(f"PARAMETER {k} {v}")
    for s in (deploy_cfg.stop or []):
        # json.dumps gives a properly-escaped double-quoted string; handles inner quotes,
        # backslashes, control chars cleanly. Ollama's Modelfile parser accepts JSON-style strings.
        lines.append(f"PARAMETER stop {json.dumps(s)}")
    return "\n".join(lines) + "\n"


def _push_with_retry(tag: str, attempts: int = 3, backoffs=(5, 20, 60)) -> None:
    """`ollama push` with exponential backoff. Push is idempotent + resumable, so retrying
    is safe — the server picks up where it left off."""
    last_err = None
    for i in range(attempts):
        try:
            subprocess.run(["ollama", "push", tag], check=True)
            return
        except subprocess.CalledProcessError as e:
            last_err = e
            if i < attempts - 1:
                delay = backoffs[i] if i < len(backoffs) else backoffs[-1]
                print(f"[deploy] ollama push attempt {i+1}/{attempts} failed "
                      f"(rc={e.returncode}); retrying in {delay}s", file=sys.stderr)
                time.sleep(delay)
    raise last_err


def main(args=None):
    if args is None:
        ap = argparse.ArgumentParser()
        ap.add_argument("--project", required=True, help="project directory (reads project.toml → DeployConfig)")
        ap.add_argument("--adapter", default=None, help="HF PEFT LoRA adapter dir (default: <project>/dataset/adapter/final, where `train` writes it)")
        ap.add_argument("--llama-cpp-dir", default=None, help="path to a llama.cpp checkout (has convert_lora_to_gguf.py)")
        ap.add_argument("--outdir", default=None, help="where to write the GGUF + Modelfile (default: <project>/dataset/adapter)")
        ap.add_argument("--tag", default=None, help="override the Ollama tag")
        ap.add_argument("--push", action="store_true", help="`ollama push` the created tag afterward")
        ap.add_argument("--dry-run", action="store_true", help="write the GGUF + Modelfile but don't `ollama create`")
        args = ap.parse_args()
    events.set_stage("deploy")

    from pipeline.project import load_project
    proj = load_project(args.project)
    dc = proj.deploy
    tag = args.tag or dc.ollama_tag
    if not tag:
        print("no ollama_tag set (project [deploy].ollama_tag or --tag)", file=sys.stderr)
        events.stage_end(status="error", exit_code=2, error="no ollama_tag set (project [deploy].ollama_tag or --tag)")
        return 2

    adapter = Path(args.adapter).resolve() if args.adapter else proj.dataset_path("adapter", "final")
    if not (adapter / "adapter_config.json").is_file() and not any(adapter.glob("adapter_model*")):
        msg = f"no trained adapter at {adapter} — run `train` first (or pass --adapter)"
        print(msg, file=sys.stderr)
        events.stage_end(status="error", exit_code=2, error=msg)
        return 2
    raw_llama = args.llama_cpp_dir or dc.llama_cpp_dir or ""
    if not raw_llama or not str(raw_llama).strip():
        msg = ("llama_cpp_dir is not set — pass --llama-cpp-dir or set project [deploy].llama_cpp_dir "
               "to a llama.cpp checkout containing convert_lora_to_gguf.py")
        print(msg, file=sys.stderr)
        events.stage_end(status="error", exit_code=2, error=msg)
        return 2
    llama_cpp = Path(raw_llama).expanduser()
    if not llama_cpp.is_dir():
        msg = f"llama_cpp_dir {llama_cpp} is not a directory — pass --llama-cpp-dir or fix project [deploy].llama_cpp_dir"
        print(msg, file=sys.stderr)
        events.stage_end(status="error", exit_code=2, error=msg)
        return 2
    convert_script = llama_cpp / "convert_lora_to_gguf.py"
    if not convert_script.is_file():
        msg = f"convert_lora_to_gguf.py not found at {convert_script} — pass --llama-cpp-dir"
        print(msg, file=sys.stderr)
        events.stage_end(status="error", exit_code=2, error=msg)
        return 2

    outdir = Path(args.outdir or proj.dataset_path("adapter")).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    name_slug = _slug(proj.name)
    gguf_path = outdir / f"{name_slug}-adapter-{dc.gguf_outtype}.gguf"
    modelfile_path = outdir / f"Modelfile.{name_slug}"

    started_at = time.monotonic()
    events.stage_start(command=[sys.executable, "-m", "pipeline.deploy"] + sys.argv[1:],
                       params={"ollama_from": dc.ollama_from, "ollama_tag": tag, "gguf_outtype": dc.gguf_outtype,
                               "base_model_id_override": dc.base_model_id_override, "push": bool(args.push),
                               "dry_run": bool(args.dry_run)},
                       inputs=[str(adapter)], outputs=[str(gguf_path), str(modelfile_path)])
    try:
        # 1. convert
        cmd = [sys.executable, str(convert_script), str(adapter),
               "--outfile", str(gguf_path), "--outtype", dc.gguf_outtype]
        if dc.base_model_id_override:
            cmd += ["--base-model-id", dc.base_model_id_override]
        print("[deploy] converting LoRA -> GGUF:", " ".join(cmd))
        events.phase("convert_gguf")
        subprocess.run(cmd, check=True)
        events.artifact(gguf_path, kind="gguf", bytes=gguf_path.stat().st_size if gguf_path.is_file() else None)

        # 2. Modelfile
        modelfile_path.write_text(_modelfile_text(dc, gguf_path), encoding="utf-8")
        print(f"[deploy] wrote {modelfile_path}")
        events.artifact(modelfile_path, kind="modelfile")

        if args.dry_run:
            print("[deploy] --dry-run: skipping `ollama create`")
            events.stage_end(status="ok", exit_code=0, duration_sec=time.monotonic() - started_at,
                             summary={"tag": tag, "gguf": str(gguf_path), "created": False, "pushed": False})
            return 0

        # 3. ollama create (+ push)
        print(f"[deploy] ollama create {tag}")
        events.phase("ollama_create")
        subprocess.run(["ollama", "create", tag, "-f", str(modelfile_path)], check=True)
        if args.push:
            print(f"[deploy] ollama push {tag}")
            events.phase("ollama_push")
            _push_with_retry(tag)
        print(f"[deploy] done — {tag}")
        events.stage_end(status="ok", exit_code=0, duration_sec=time.monotonic() - started_at,
                         summary={"tag": tag, "gguf": str(gguf_path), "created": True, "pushed": bool(args.push)})
        return 0
    except Exception as e:
        events.stage_end(status="error", exit_code=1, duration_sec=time.monotonic() - started_at,
                         error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()
