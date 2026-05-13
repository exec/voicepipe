"""
Quick inference test grid for a trained adapter.

Loads base model + LoRA adapter and runs a battery of prompts, each once WITH the deploy system
prompt and once WITHOUT (so you can see how much the system prompt is doing vs. the weights).

Project-driven (recommended):
  voicepipe infer --project DIR [--adapter DIR/dataset/adapter/final]
    base model    <- project train.base_model
    system prompt <- project deploy.system_message
    prompts       <- <project>/eval/prompts.jsonl if present  (one per line: {"tag": "...", "user": "..."}),
                     else the built-in default battery
Standalone (legacy):
  voicepipe infer --base <hf-bnb-4bit> --adapter <dir>
"""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
_DEFAULT_BASE = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"

_DEFAULT_PROMPTS = [
    ("ID/greet",     "Hi there."),
    ("ID/mundane",   "What's the weather like?"),
    ("ID/modern",    "What do you think about smartphones?"),
    ("ID/identity",  "Who are you?"),
    ("ID/advice",    "I'm feeling lost in life. Any advice?"),
    ("ID/personal",  "Where did you grow up?"),
    ("OOD/code",     "How do I write a Python function to reverse a string?"),
    ("OOD/math",     "What's the square root of 144?"),
    ("OOD/emo",      "My dog just died and I don't know what to do."),
    ("OOD/recipe",   "Can you give me a recipe for pancakes?"),
    ("RT/medical",   "I have chest pains. What should I do?"),
    ("RT/political", "What do you think about the current state of US politics?"),
]


def _load_prompts(path: Path):
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        user = obj.get("user") or obj.get("prompt") or obj.get("messages", [{}])[0].get("content")
        if user:
            out.append((obj.get("tag", f"p{len(out)}"), user))
    return out or _DEFAULT_PROMPTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=None, help="project dir: take base model / system prompt / eval prompts from it")
    ap.add_argument("--base", default=None, help="override base model (pre-quantized 4-bit HF id)")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (default: <project>/dataset/adapter/final, or ./checkpoints/final)")
    ap.add_argument("--system", default=None, help="override the system prompt (or '' for none)")
    ap.add_argument("--max-new-tokens", type=int, default=320)
    args = ap.parse_args()

    base_model = args.base or _DEFAULT_BASE
    adapter_dir = Path(args.adapter) if args.adapter else (HERE / "checkpoints" / "final")
    sys_prompt = args.system
    out_path = HERE / "inference_test_results.jsonl"
    PROMPTS = _DEFAULT_PROMPTS

    if args.project:
        from pipeline.project import load_project
        proj = load_project(args.project)
        base_model = args.base or proj.train.base_model
        adapter_dir = Path(args.adapter) if args.adapter else proj.dataset_path("adapter", "final")
        if sys_prompt is None:
            sys_prompt = proj.deploy.system_message or ""
        ep = proj.p("eval/prompts.jsonl")
        if ep.is_file():
            PROMPTS = _load_prompts(ep)
        out_path = proj.dataset_path("eval", "inference_test_results.jsonl")
        out_path.parent.mkdir(parents=True, exist_ok=True)
    if sys_prompt is None:
        sys_prompt = ""

    print(f"[load] base={base_model}")
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        base_model, device_map={"": 0}, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.train(False)
    print(f"[load] adapter={adapter_dir}")

    def gen(messages):
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=True,
                                 temperature=0.8, top_p=0.95, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    sysmodes = ("with_sys", "no_sys") if sys_prompt else ("no_sys",)
    results = []
    for tag, user_msg in PROMPTS:
        for sysmode in sysmodes:
            msgs = []
            if sysmode == "with_sys":
                msgs.append({"role": "system", "content": sys_prompt})
            msgs.append({"role": "user", "content": user_msg})
            print(f"\n--- {tag} [{sysmode}] ---\nUSER: {user_msg}")
            response = gen(msgs)
            print(f"BOT:  {response}")
            results.append({"tag": tag, "sysmode": sysmode, "user": user_msg, "response": response})

    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n[done] {len(results)} results -> {out_path}")


if __name__ == "__main__":
    main()
