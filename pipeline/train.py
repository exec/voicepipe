"""
QLoRA fine-tune a base model on an assembled dataset.

Config comes from a project (`--project DIR` reads project.toml → TrainConfig + dataset paths);
every TrainConfig field is also an explicit CLI flag, so you can override or run standalone.

Handles three base-model shapes:
  1. plain CausalLM checkpoints (Llama, Mistral, Mistral Nemo, ...)
  2. multimodal vision+text checkpoints — keeps the ...ForConditionalGeneration wrapper so
     peft+SFTTrainer have an lm head; the LoRA target_modules only match the language layers
  3. unsloth FastLanguageModel fallback (handles bnb-4bit + multimodal cleanly)

Run on the training box (a CUDA GPU). The `pipeline` package must be importable
(`pip install -e .` from the repo root, or PYTHONPATH).

Usage:
  python -m pipeline.train --project scratch/dec-bot                # full run, config from project.toml
  python -m pipeline.train --project scratch/dec-bot --smoke        # 10-step dry run
  python -m pipeline.train --project scratch/dec-bot --resume-adapter <dir>   # continue an existing adapter
  python -m pipeline.train --model unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit --train-jsonl data/train.jsonl ...

Two GPUs, two trainings at once (one per card):
  python -m pipeline.train --gpu 0 --project scratch/dec-bot          &
  python -m pipeline.train --gpu 1 --project projects/other-character &
  wait
(Each --gpu N process sees only that physical card; it appears as device 0 internally. Always
pass --gpu when running more than one trainer on a multi-GPU box, or both land on GPU 0 and OOM.)
"""

import argparse
import json
import os
import sys
from pathlib import Path

# On a multi-GPU box, pin this process to one card BEFORE torch initializes CUDA.
# `--gpu N` (or the CUDA_VISIBLE_DEVICES env var) controls which physical GPU is used; inside
# the process it always appears as device 0, which is what device_map={"": 0} targets.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--gpu", type=int, default=None)
_known, _ = _pre.parse_known_args()
if _known.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_known.gpu)

import time
import traceback

from pipeline import events

# Heavy ML deps — only needed to actually run training, not to render `--help`. Deferred
# behind a guard so `voicepipe train --help` works without the [train] extra installed.
# `_EventsCallback` and `_DTYPE_KWARG` reference imported symbols at definition time, so they
# live inside the guard too; the stage's functions reference these names only at call time.
try:
    import torch
    import transformers
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
    from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer
    _HEAVY_OK = True
except ImportError:
    _HEAVY_OK = False

if _HEAVY_OK:
    class _EventsCallback(TrainerCallback):
        """Emits a `metric` + `progress` event on every trainer log step."""
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            numeric = {k: v for k, v in logs.items() if isinstance(v, (int, float)) and k not in ("epoch", "step")}
            events.metric(step=state.global_step, epoch=round(state.epoch, 4) if state.epoch is not None else None, **numeric)
            events.progress(current=state.global_step, total=(state.max_steps or None), unit="steps")

    # transformers 5.x renamed the from_pretrained dtype kwarg: torch_dtype -> dtype.
    # Brittle for dev/RC versions ("5.0.0.dev0") but split-and-int handles those; truly exotic
    # version strings would raise — that's an acceptable failure mode for a build-time pin.
    _DTYPE_KWARG = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"


def _is_quantized_4bit(model) -> bool:
    """Return True if `model` actually has bitsandbytes 4-bit layers installed."""
    qc = getattr(getattr(model, "config", None), "quantization_config", None)
    if qc is not None:
        if getattr(qc, "load_in_4bit", False):
            return True
        # dict form on some transformers versions
        if isinstance(qc, dict) and qc.get("load_in_4bit"):
            return True
    cls_names = {type(m).__name__ for _, m in model.named_modules()}
    return "Linear4bit" in cls_names


def _load_jsonl(p: Path):
    return [json.loads(line) for line in Path(p).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_base_model(model_name: str):
    """Load a 4-bit base model for QLoRA. See module docstring for the three shapes handled.

    The primary (transformers) path passes an explicit BitsAndBytesConfig so the model is
    quantized at load time even when `model_name` is not a pre-quantized repo (e.g. someone
    passes `mistralai/Mistral-Nemo-Instruct-2407` instead of the `unsloth/...-bnb-4bit` mirror).
    Without this, that mistake silently loads full bf16 and OOMs.
    """
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    primary_err = None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map={"": 0}, quantization_config=bnb,
            **{_DTYPE_KWARG: torch.bfloat16}, trust_remote_code=True,
        )
        print(f"[model] loaded {model_name} via AutoModelForCausalLM")
        if not _is_quantized_4bit(model):
            raise RuntimeError(
                f"loaded {model_name} but the model has no bitsandbytes 4-bit layers — "
                f"bitsandbytes import probably failed silently, or the model architecture "
                f"isn't supported by bnb. Refusing to continue (full-precision training would OOM)."
            )
        return model
    except (ValueError, KeyError, OSError, RuntimeError) as e:
        primary_err = e
        print(f"[model] AutoModelForCausalLM failed ({type(e).__name__}: {e}); trying multimodal path")

    try:
        from transformers import AutoModelForImageTextToText
        mm = AutoModelForImageTextToText.from_pretrained(
            model_name, device_map={"": 0}, quantization_config=bnb,
            **{_DTYPE_KWARG: torch.bfloat16}, trust_remote_code=True,
        )
        print(f"[model] loaded {model_name} via AutoModelForImageTextToText (full multimodal wrapper)")
        if not _is_quantized_4bit(mm):
            raise RuntimeError(
                f"loaded multimodal {model_name} but no bitsandbytes 4-bit layers present; "
                f"refusing to continue."
            )
        return mm
    except Exception as e:
        print(f"[model] multimodal path also failed ({type(e).__name__}: {e})")

    try:
        from unsloth import FastLanguageModel
        model, _tok = FastLanguageModel.from_pretrained(
            model_name=model_name, max_seq_length=4096, dtype=torch.bfloat16, load_in_4bit=True,
        )
        print(f"[model] loaded {model_name} via unsloth FastLanguageModel")
        return model
    except ImportError:
        print("[model] unsloth not installed; cannot use FastLanguageModel fallback")
    except Exception as e:
        print(f"[model] unsloth path also failed ({type(e).__name__}: {e})")

    raise RuntimeError(
        f"Could not load {model_name} via any path (primary error: {primary_err}). "
        f"If this is a multimodal model, `pip install unsloth` in the venv and rerun."
    )


def _resolve_config(args):
    """Merge: explicit CLI flags > project.toml TrainConfig > dataclass defaults.
    Returns a dict of the effective settings + the resolved train/val jsonl paths + output dir."""
    from pipeline.project import TrainConfig
    if args.project:
        from pipeline.project import load_project
        proj = load_project(args.project)
        tc = proj.train
        train_jsonl = args.train_jsonl or proj.dataset_path("final", "train.jsonl")
        val_jsonl = args.val_jsonl or proj.dataset_path("final", "val.jsonl")
        out_dir = args.out_dir or proj.dataset_path("adapter")
    else:
        tc = TrainConfig()
        train_jsonl = Path(args.train_jsonl or "data/train.jsonl")
        val_jsonl = Path(args.val_jsonl or "data/val.jsonl")
        out_dir = Path(args.out_dir or "checkpoints")

    def pick(cli_val, cfg_attr):
        return cli_val if cli_val is not None else getattr(tc, cfg_attr)

    return {
        "model": pick(args.model, "base_model"),
        "lora_r": pick(args.lora_r, "lora_r"),
        "lora_alpha": pick(args.lora_alpha, "lora_alpha"),
        "lora_dropout": pick(args.lora_dropout, "lora_dropout"),
        "target_modules": tc.target_modules,
        "max_seq_len": pick(args.max_seq_len, "max_seq_len"),
        "batch_size": pick(args.batch_size, "batch_size"),
        "grad_accum": pick(args.grad_accum, "grad_accum"),
        "epochs": pick(args.epochs, "epochs"),
        "lr": pick(args.lr, "lr"),
        "optim": pick(args.optim, "optim"),
        "seed": tc.seed,
        "train_jsonl": Path(train_jsonl),
        "val_jsonl": Path(val_jsonl),
        "out_dir": Path(out_dir),
    }


def main(args=None):
    if args is None:
        ap = argparse.ArgumentParser()
        ap.add_argument("--gpu", type=int, default=None, help="pin to this physical GPU (sets CUDA_VISIBLE_DEVICES); for running two trainings on a 2-GPU box")
        ap.add_argument("--project", default=None, help="project directory (reads project.toml → TrainConfig + dataset paths)")
        ap.add_argument("--model", default=None, help="override: pre-quantized 4-bit base model on HF Hub")
        ap.add_argument("--train-jsonl", default=None, help="override path to train.jsonl")
        ap.add_argument("--val-jsonl", default=None, help="override path to val.jsonl")
        ap.add_argument("--out-dir", default=None, help="override adapter output directory")
        ap.add_argument("--epochs", type=float, default=None)
        ap.add_argument("--batch-size", type=int, default=None)
        ap.add_argument("--grad-accum", type=int, default=None)
        ap.add_argument("--lr", type=float, default=None)
        ap.add_argument("--lora-r", type=int, default=None)
        ap.add_argument("--lora-alpha", type=int, default=None)
        ap.add_argument("--lora-dropout", type=float, default=None)
        ap.add_argument("--max-seq-len", type=int, default=None)
        ap.add_argument("--optim", default=None)
        ap.add_argument("--resume-adapter", default=None, help="continue training an existing LoRA adapter dir (weights only — discards optimizer/scheduler/RNG state)")
        ap.add_argument("--resume-from-checkpoint", default=None,
                        help="resume optimizer+scheduler+RNG state from a Trainer checkpoint dir; "
                             "pass 'auto' to pick the latest out_dir/checkpoint-* automatically")
        ap.add_argument("--smoke", action="store_true", help="10-step dry run: tiny subset, verifies load + GPU + pipeline")
        args = ap.parse_args()

    if not _HEAVY_OK:
        raise SystemExit("voicepipe train needs the training extra — `pip install -e '.[train]'` "
                         "(plus torch from the PyTorch CUDA index; see constraints-train.txt).")

    events.set_stage("train")

    cfg = _resolve_config(args)
    started_at = time.monotonic()

    print(f"[torch] {torch.__version__} cuda={torch.version.cuda}")
    print(f"[gpu]   {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_capability(0)})")
    free, total = torch.cuda.mem_get_info(0)
    print(f"[vram]  {free/1e9:.1f} / {total/1e9:.1f} GB free")
    print(f"[cfg]   model={cfg['model']} r={cfg['lora_r']} alpha={cfg['lora_alpha']} "
          f"seq={cfg['max_seq_len']} bs={cfg['batch_size']} ga={cfg['grad_accum']} epochs={cfg['epochs']}")
    print(f"[data]  train={cfg['train_jsonl']}  val={cfg['val_jsonl']}  -> adapter={cfg['out_dir']}")
    events.stage_start(
        command=[sys.executable, "-m", "pipeline.train"] + sys.argv[1:],
        params={"base_model": cfg["model"], "lora_r": cfg["lora_r"], "lora_alpha": cfg["lora_alpha"],
                "lora_dropout": cfg["lora_dropout"], "max_seq_len": cfg["max_seq_len"],
                "batch_size": cfg["batch_size"], "grad_accum": cfg["grad_accum"], "epochs": cfg["epochs"],
                "lr": cfg["lr"], "optim": cfg["optim"], "smoke": bool(args.smoke),
                "gpu": args.gpu, "resume_adapter": args.resume_adapter,
                "resume_from_checkpoint": args.resume_from_checkpoint,
                "device": torch.cuda.get_device_name(0)},
        inputs=[str(cfg["train_jsonl"]), str(cfg["val_jsonl"])], outputs=[str(cfg["out_dir"])])

    try:
        # Ministral 3 / Mistral-Small tokenizers ship a regex transformers 5.x flags; the kwarg fixes it
        # (harmless / ignored on older transformers and non-Mistral tokenizers).
        try:
            tok = AutoTokenizer.from_pretrained(cfg["model"], trust_remote_code=True, fix_mistral_regex=True)
        except TypeError:
            tok = AutoTokenizer.from_pretrained(cfg["model"], trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        events.phase("load_base_model")
        model = load_base_model(cfg["model"])
        model = prepare_model_for_kbit_training(model)

        events.phase("build_peft")
        if args.resume_adapter:
            model = PeftModel.from_pretrained(model, args.resume_adapter, is_trainable=True)
            print(f"[lora]  resumed from existing adapter: {args.resume_adapter}")
        else:
            model = get_peft_model(model, LoraConfig(
                r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
                target_modules=cfg["target_modules"], bias="none", task_type="CAUSAL_LM",
            ))
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_p = sum(p.numel() for p in model.parameters())
        print(f"[lora]  trainable params: {trainable/1e6:.1f}M / {total_p/1e9:.2f}B ({100*trainable/total_p:.2f}%)")
        events.log(f"trainable params: {trainable/1e6:.1f}M / {total_p/1e9:.2f}B ({100*trainable/total_p:.2f}%)")

        def to_text(example):
            return {"text": tok.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)}

        train_rows = _load_jsonl(cfg["train_jsonl"])
        val_rows = _load_jsonl(cfg["val_jsonl"])
        if not train_rows:
            raise RuntimeError(
                f"no training rows in {cfg['train_jsonl']} — did `assemble` run? "
                f"(expected one {{\"messages\": [...]}} per line)"
            )
        if args.smoke:
            train_rows, val_rows = train_rows[:20], val_rows[:5]
        print(f"[data]  loaded train={len(train_rows)} val={len(val_rows)}")

        train_ds = Dataset.from_list(train_rows).map(to_text, remove_columns=list(train_rows[0].keys()))
        val_ds = (Dataset.from_list(val_rows).map(to_text, remove_columns=list(val_rows[0].keys()))
                  if val_rows else None)

        # Label-mask everything up to (and including) the assistant header so loss is computed
        # only on assistant tokens. Mistral / Mistral-Nemo's chat template ends the user turn
        # with `[/INST]`; we look for that exact string in the templated text. For other
        # tokenizers we try a small set of well-known headers; if none match, SFTTrainer would
        # silently compute loss on prompt tokens too — so we fail loudly instead.
        response_template = None
        sample_text = train_ds[0]["text"]
        for candidate in ("[/INST]", "<|start_header_id|>assistant<|end_header_id|>\n\n",
                          "<|im_start|>assistant\n", "<start_of_turn>model\n"):
            if candidate in sample_text:
                response_template = candidate
                break
        if response_template is None:
            raise RuntimeError(
                "could not locate an assistant-turn boundary in the chat-templated text "
                "(tried Mistral [/INST], Llama 3, ChatML, Gemma). Add this tokenizer's "
                "response template to pipeline/train.py."
            )
        collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tok)
        print(f"[mask]  response_template={response_template!r} (loss on assistant tokens only)")

        # bf16 requires Ampere or newer (sm_80+). On Turing/Volta/T4/P100, fall back to fp16.
        # Mutually-exclusive: TrainingArguments rejects both.
        use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        use_fp16 = bool(torch.cuda.is_available() and not use_bf16)
        print(f"[prec]  bf16={use_bf16} fp16={use_fp16}")

        out_dir = cfg["out_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)

        # Pick a resume-from-checkpoint target: explicit path, 'auto' (latest checkpoint-*), or None.
        resume_target = None
        if args.resume_from_checkpoint:
            if args.resume_from_checkpoint == "auto":
                ckpts = sorted(out_dir.glob("checkpoint-*"),
                               key=lambda p: int(p.name.rsplit("-", 1)[-1]) if p.name.rsplit("-", 1)[-1].isdigit() else -1)
                resume_target = str(ckpts[-1]) if ckpts else None
                if resume_target:
                    print(f"[resume] auto-detected latest checkpoint: {resume_target}")
                else:
                    print(f"[resume] --resume-from-checkpoint=auto: no checkpoint-* in {out_dir}, starting fresh")
            else:
                resume_target = args.resume_from_checkpoint

        sft = SFTConfig(
            output_dir=str(out_dir),
            num_train_epochs=1 if args.smoke else cfg["epochs"],
            max_steps=10 if args.smoke else -1,
            per_device_train_batch_size=cfg["batch_size"],
            per_device_eval_batch_size=cfg["batch_size"],
            gradient_accumulation_steps=cfg["grad_accum"],
            learning_rate=cfg["lr"],
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            max_seq_length=cfg["max_seq_len"],
            packing=False,
            dataset_text_field="text",
            logging_steps=5,
            save_strategy="epoch" if not args.smoke else "no",
            eval_strategy=("epoch" if (not args.smoke and val_ds is not None) else "no"),
            bf16=use_bf16,
            fp16=use_fp16,
            gradient_checkpointing=True,
            optim=cfg["optim"],
            report_to="none",
            seed=cfg["seed"],
        )
        trainer = SFTTrainer(model=model, args=sft, train_dataset=train_ds, eval_dataset=val_ds,
                             processing_class=tok, data_collator=collator,
                             callbacks=[_EventsCallback()])

        print("[train] starting..." + (f" (resume_from_checkpoint={resume_target})" if resume_target else ""))
        events.phase("train")
        train_result = trainer.train(resume_from_checkpoint=resume_target) if resume_target else trainer.train()

        if not args.smoke:
            events.phase("save")
            final_dir = out_dir / "final"
            trainer.save_model(str(final_dir))
            tok.save_pretrained(str(final_dir))
            print(f"[done] adapter saved to {final_dir}")
            events.artifact(final_dir, kind="adapter")
        else:
            print("[done] smoke test complete — pipeline works")

        m = getattr(train_result, "metrics", {}) or {}
        events.stage_end(status="ok", exit_code=0, duration_sec=time.monotonic() - started_at,
                         summary={"steps": getattr(trainer.state, "global_step", None),
                                  "train_loss": m.get("train_loss"), "smoke": bool(args.smoke),
                                  "adapter_dir": None if args.smoke else str(out_dir / "final")})
    except Exception as e:
        events.stage_end(status="error", exit_code=1, duration_sec=time.monotonic() - started_at,
                         error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()
