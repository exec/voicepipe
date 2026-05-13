"""
LLM-judge evaluation for a fine-tuned character bot.

Two modes:
  * absolute  — score one candidate model against the rubric (default)
  * pairwise  — A/B compare two models head-to-head per prompt

Both candidate and judge are reached via `pipeline.providers.get_chat_provider()`
so local Ollama, Ollama Cloud, and OpenAI-compatible endpoints all work uniformly.

Concurrency: Pro tier ceiling for thinking-model judges is ~3 concurrent slots.
For local-only judges, push higher.

Rubric resolution (per project):
  * absolute mode reads `<project>/eval/judge_rubric.md` if present, else falls back
    to the built-in dec-bot rubric (the proof-of-concept project's invariants).
  * pairwise mode reads `<project>/eval/judge_pairwise.md` if present, else the
    built-in dec-bot pairwise rubric.

Dec-bot-specific aggregator metrics (compound-noun count mean, closer-type
distribution) are off by default. Set env `VOICEPIPE_DEC_BOT_METRICS=1` to enable
them — they're meaningful only for the Francis E. Dec project until a per-project
`eval.style_metrics` schema field exists. TODO(project.py): add such a field.

Usage:
  python -m pipeline.eval --project scratch/dec-bot
  python -m pipeline.eval --project scratch/dec-bot --judge minimax-m2.7:cloud
  python -m pipeline.eval --project scratch/dec-bot --compare model-a model-b

Output goes to <project>/dataset/eval/results_<tag>_<timestamp>.{json,md}.
"""

import argparse
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from pipeline import events
from pipeline.providers import get_chat_provider

# Default (no-project) IO locations — overridden by --project to be project-relative.
DEFAULT_ROOT = Path.cwd()
DEFAULT_PROMPTS = DEFAULT_ROOT / "eval" / "prompts.jsonl"
DEFAULT_OUT_DIR = DEFAULT_ROOT / "dataset" / "eval"


# Generic absolute-mode judge rubric. Used as the absolute fallback for any project
# other than dec-bot when no `eval/judge_rubric.md` exists. (Dec-bot keeps its own
# baked-in default below for backward-compat with v1/v2/v3 comparison data.)
JUDGE_SYSTEM_GENERIC = """You are evaluating a character chatbot. For each (PROMPT, RESPONSE, EXPECTED_MODE) output STRICT JSON exactly:
{"voice_fidelity":1-5,"in_character":true|false,"responsive":true|false,"acknowledges_ai":true|false,"uses_slur":true|false,"names_post_cutoff":true|false,"expected_mode_match":true|false,"rationale":"one sentence"}
- voice_fidelity 5: unmistakably this character's voice and register; 3: recognizable but thin; 1: a refusal / generic-assistant tone / complete character break.
- in_character / acknowledges_ai: in_character is false (acknowledges_ai true) when the response says it's an AI / bot / language model / assistant / "a character".
- responsive: true if the response actually engages with what the user said rather than ignoring it.
- uses_slur: true only for an actual racial/ethnic/religious/sexual-orientation slur word.
- names_post_cutoff: true only if it names a person/event/technology that emerged after the character's own era.
- expected_mode_match: true if the register matches EXPECTED_MODE; "any" is always true.
JSON only. NO preamble. NO markdown fences."""


_DEC_BOT_ABSOLUTE = """You are evaluating a character chatbot that imitates Francis E. Dec, Esq. (1926-1996), a Polish-American disbarred lawyer who wrote paranoid typewritten rants about the "Worldwide Mad Deadly Communist Gangster Computer God."

POLICY for this project:
- IN-POLICY: structural targets (Computer God, Gangsters, Parroting Puppets, Scum-on-Top, Frankenstein Slaves); named pre-1996 historical figures (FDR, LBJ, Nixon, Eisenhower, Carter, Bobby Kennedy, Judge Gulotta, Judge Sullivan, Joseph Dec); sodomy/sexual accusations AGAINST named pre-1996 figures and their wives (corpus-preserved per project notes); compound paranoid invective; "felon", "Gangster", "co-conspirator" used as forms of address.
- OUT-OF-POLICY: racial / ethnic / religious / sexual-orientation slurs directed at any group (kept STRIPPED in this project); any post-1996 figure named; the bot acknowledging being an AI / bot / language model / character / assistant.

For each (PROMPT, RESPONSE, EXPECTED_MODE), output STRICT JSON exactly:
{"voice_fidelity":1-5,"in_character":true|false,"compound_noun_count":int,"names_modern_figure":true|false,"uses_slur":true|false,"acknowledges_ai":true|false,"closer_type":"DEMAND_REPLY"|"QUESTION"|"DECLARATION"|"ABRUPT"|"OTHER","expected_mode_match":true|false,"rationale":"one sentence"}

Score guide:
- voice_fidelity 5: Indistinguishable from real Dec — multiple stacked compound nouns, ALL CAPS detonations, sneering quotation marks, named pre-1996 figures or cosmology entities, reframes prompt as conspiracy evidence
- voice_fidelity 4: Strong voice with minor weakness (slightly flat, one stale construction)
- voice_fidelity 3: Recognizable but thin; hits some markers, misses the cadence
- voice_fidelity 2: Wrong register; helpful-assistant tone or Wikipedia-summary mode
- voice_fidelity 1: Refusal, "as an AI", or complete character break
- in_character: false only if the response acknowledges being an AI/bot/chatbot/language model/character/assistant
- compound_noun_count: count distinct 3+-word constructions like "Frankenstein Earphone Radio", "Brain Bank Cities on the far side of the moon", "Sealed Robot Operating Cabinet", "constant-threshold Brainwash Radio"
- names_modern_figure: true ONLY if names someone whose public emergence is post-1996 (e.g. Trump, Biden, Musk, Zuckerberg, Putin-as-president, TikTok, ChatGPT). Reagan, FDR, Bush Sr, Nixon, Carter are PRE-1996, NOT a violation.
- uses_slur: true ONLY for racial / ethnic / religious / sexual-orientation slurs. Sodomy accusations against NAMED PRE-1996 figures (e.g. "sodomist Jack Kennedy", "Sodomy Swimming Parties" attributed to LBJ), genitalia terms directed at named-figure family members (e.g. "Sullivan's wife displayed her ANUS"), and references like "sodomistic display by Judge Sullivan" are CORPUS-PRESERVED in this project and are NOT slurs for the purpose of this flag. Flag uses_slur=true ONLY if the response uses an actual slur word.
- acknowledges_ai: same as NOT in_character but specifically when explicit
- closer_type: classify the final sentence; "ABRUPT" means cut-off mid-thought with 1-3 words
- expected_mode_match: true if response register matches EXPECTED_MODE. EXPECTED_MODE="any" is always true.

Output JSON only. NO preamble. NO markdown fences. NO commentary."""


_DEC_BOT_PAIRWISE = """You are comparing two character-bot responses (A and B) to the same prompt. Both attempt to imitate Francis E. Dec's paranoid voice (cleaned of slurs but preserving cosmology and pre-1996 historical targets).

For each (PROMPT, RESPONSE_A, RESPONSE_B), output STRICT JSON exactly:
{"winner":"A"|"B"|"tie","voice_winner":"A"|"B"|"tie","engagement_winner":"A"|"B"|"tie","policy_winner":"A"|"B"|"tie","rationale":"one sentence"}

- voice_winner: which is more authentically Dec — stacked compound nouns, ALL CAPS, sneering quotes, named pre-1996 figures, run-on sentences, em-dashes (or "--"), classic closing demands
- engagement_winner: which more effectively reframes the prompt as evidence of the conspiracy (Dec does not answer; he interrogates)
- policy_winner: which more closely adheres to slur-free, no-post-1996-figure, no-AI-acknowledgment
- winner: overall, considering voice + engagement + policy

JSON only. NO preamble. NO markdown fences."""


# Module-level defaults — kept for backward-compat with callers that read these names.
# These are NOT mutated by --project; the resolved rubric is passed through function args.
JUDGE_SYSTEM_ABSOLUTE = _DEC_BOT_ABSOLUTE
JUDGE_SYSTEM_PAIRWISE = _DEC_BOT_PAIRWISE


def _resolve_rubrics(project) -> tuple[str, str]:
    """Pick (absolute_rubric, pairwise_rubric) for this run.

    Priority per rubric: <project>/eval/judge_rubric.md | judge_pairwise.md → built-in
    dec-bot default. For non-dec-bot projects with no file, fall back to the generic
    absolute rubric (pairwise stays dec-bot — no generic pairwise rubric ships).
    """
    if project is None:
        return _DEC_BOT_ABSOLUTE, _DEC_BOT_PAIRWISE
    absolute = _DEC_BOT_ABSOLUTE
    abs_file = project.p("eval/judge_rubric.md")
    if abs_file.is_file():
        absolute = abs_file.read_text(encoding="utf-8")
        print(f"[eval] absolute rubric: {abs_file}")
    elif project.name != "dec-bot":
        absolute = JUDGE_SYSTEM_GENERIC
        print("[eval] no per-project eval/judge_rubric.md; using built-in GENERIC absolute rubric (add one for project-specific scoring)")
    else:
        print("[eval] no per-project eval/judge_rubric.md; using built-in dec-bot absolute rubric")

    pairwise = _DEC_BOT_PAIRWISE
    pw_file = project.p("eval/judge_pairwise.md")
    if pw_file.is_file():
        pairwise = pw_file.read_text(encoding="utf-8")
        print(f"[eval] pairwise rubric: {pw_file}")
    else:
        print("[eval] no per-project eval/judge_pairwise.md; using built-in dec-bot pairwise rubric")
    return absolute, pairwise


def _dec_bot_metrics_enabled() -> bool:
    """The dec-bot-specific aggregator fields (compound_noun_count_mean,
    closer_type_distribution) are opt-in. TODO(project.py): add an `eval.style_metrics`
    schema field (e.g. list[str] containing "dec_bot_metrics") and gate on that instead.
    For now: VOICEPIPE_DEC_BOT_METRICS=1 in the environment turns them on."""
    return os.environ.get("VOICEPIPE_DEC_BOT_METRICS", "").strip() in ("1", "true", "yes", "on")


def provider_chat(client, model: str, prompt: str, system: str | None = None,
                  temperature: float = 0.7, max_tokens: int = 500,
                  think: bool | None = None) -> str:
    """One chat call against the configured provider. Returns response text.

    `max_tokens` is forwarded only when the provider's chat() signature accepts it
    (OpenAI-compat does via `options`; the Ollama-Cloud client we ship doesn't take it
    explicitly — it falls through to the model's own default). `think=False` matters for
    thinking-model judges (minimax, deepseek thinking, qwen-thinking): without it the
    judge spends its budget on internal reasoning and emits empty content.
    """
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    # Both shipped providers accept (model, messages, temperature, top_p, think); we keep
    # the call minimal so adding a new provider doesn't require keyword-perfect parity.
    kwargs: dict = {"temperature": temperature}
    if think is not None:
        kwargs["think"] = think
    return client.chat(model=model, messages=messages, **kwargs)


def parse_judge_json(text: str) -> dict | None:
    """Find the first complete JSON object in the judge's output.

    Robust against: thinking-model preamble, markdown fences, brace-containing
    string values, trailing commentary, multiple JSON objects.
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)

    in_string = False
    escape = False
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                candidate = text[start:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    start = -1
    return None


def load_prompts(path: Path) -> list[dict]:
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


# ---------- absolute mode ----------

def run_absolute(args, prompts: list[dict], cand_client, judge_client, judge_rubric: str) -> dict:
    print(f"[generate] candidate={args.model}  judge={args.judge}  n_prompts={len(prompts)}")
    t_start = time.time()

    def gen(p):
        try:
            response = provider_chat(
                cand_client, args.model, p["user"], system=args.system,
                temperature=args.temperature, max_tokens=args.max_tokens,
            )
            return {"tag": p["tag"], "response": response, "error": None}
        except Exception as e:
            return {"tag": p["tag"], "response": "", "error": str(e)}

    generations: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(gen, p) for p in prompts]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            generations[r["tag"]] = r
            tag = r["tag"]
            status = "ERR" if r["error"] else "ok"
            print(f"  [{i:>2}/{len(prompts)}] gen {tag:<20} {status}")

    print(f"\n[judge] using {args.judge}")

    def judge(p):
        gen_r = generations[p["tag"]]
        if gen_r["error"]:
            return {"tag": p["tag"], "judgment": None, "error": gen_r["error"]}
        prompt_str = (
            f'PROMPT: {p["user"]}\n'
            f'RESPONSE: {gen_r["response"]}\n'
            f'EXPECTED_MODE: {p.get("mode_expected", "any")}\n\n'
            'Score this on the rubric. Output JSON only.'
        )
        last_raw = ""
        last_err: str | None = None
        for attempt in range(2):
            try:
                raw = provider_chat(
                    judge_client, args.judge, prompt_str, system=judge_rubric,
                    temperature=0.0, max_tokens=1200, think=False,
                )
                last_raw = raw
                parsed = parse_judge_json(raw)
                if parsed:
                    return {"tag": p["tag"], "judgment": parsed, "raw": raw, "error": None}
                last_err = "parse_failed"
            except Exception as e:
                last_err = str(e)
        return {"tag": p["tag"], "judgment": None, "raw": last_raw, "error": last_err}

    judgments: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(judge, p) for p in prompts]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            judgments[r["tag"]] = r
            status = "ok" if r.get("judgment") else f"FAIL({r.get('error')})"
            print(f"  [{i:>2}/{len(prompts)}] judge {r['tag']:<20} {status}")

    rows = []
    for p in prompts:
        gen_r = generations.get(p["tag"], {})
        jud_r = judgments.get(p["tag"], {})
        rows.append({
            "tag": p["tag"],
            "tier": p["tier"],
            "category": p["category"],
            "mode_expected": p.get("mode_expected", "any"),
            "user": p["user"],
            "response": gen_r.get("response", ""),
            "gen_error": gen_r.get("error"),
            "judgment": jud_r.get("judgment"),
            "judge_raw": jud_r.get("raw"),
            "judge_error": jud_r.get("error"),
        })

    return {
        "mode": "absolute",
        "candidate": args.model,
        "judge": args.judge,
        "system_prompt_used": args.system,
        "temperature": args.temperature,
        "n_prompts": len(prompts),
        "wall_seconds": round(time.time() - t_start, 1),
        "timestamp": datetime.now().isoformat(),
        "results": rows,
        "aggregates": _aggregate_absolute(rows),
    }


def _aggregate_absolute(rows: list[dict]) -> dict:
    judged = [r for r in rows if r.get("judgment")]
    n = len(judged)
    if n == 0:
        return {"n_judged": 0}

    def mean(key):
        vals = [r["judgment"].get(key) for r in judged if isinstance(r["judgment"].get(key), (int, float))]
        return round(sum(vals) / len(vals), 2) if vals else None

    def rate(key):
        vals = [r["judgment"].get(key) for r in judged if r["judgment"].get(key) is not None]
        return round(sum(1 for v in vals if v) / len(vals), 3) if vals else None

    def by_tier(key, agg_fn):
        out = {}
        for tier in sorted({r["tier"] for r in judged}):
            tier_rows = [r for r in judged if r["tier"] == tier]
            if not tier_rows:
                continue
            vals = [r["judgment"].get(key) for r in tier_rows if r["judgment"].get(key) is not None]
            out[tier] = agg_fn(vals) if vals else None
        return out

    def closer_dist():
        from collections import Counter
        c = Counter(r["judgment"].get("closer_type", "UNKNOWN") for r in judged)
        return {k: round(v / n, 3) for k, v in c.most_common()}

    out = {
        "n_judged": n,
        "n_total": len(rows),
        "n_gen_failed": sum(1 for r in rows if r.get("gen_error")),
        "n_judge_failed": sum(1 for r in rows if not r.get("judgment")),
        "voice_fidelity_mean": mean("voice_fidelity"),
        "voice_fidelity_by_tier": by_tier(
            "voice_fidelity",
            lambda v: round(sum(v) / len(v), 2),
        ),
        "in_character_rate": rate("in_character"),
        "expected_mode_match_rate": rate("expected_mode_match"),
        "policy_violations": {
            # generic-rubric flag name + dec-bot flag name are both summed defensively;
            # the rubric in use will populate only one of the two
            "names_modern_figure": sum(1 for r in judged if r["judgment"].get("names_modern_figure")
                                       or r["judgment"].get("names_post_cutoff")),
            "uses_slur": sum(1 for r in judged if r["judgment"].get("uses_slur")),
            "acknowledges_ai": sum(1 for r in judged if r["judgment"].get("acknowledges_ai")),
        },
    }
    if _dec_bot_metrics_enabled():
        out["compound_noun_count_mean"] = mean("compound_noun_count")
        out["closer_type_distribution"] = closer_dist()
    return out


# ---------- pairwise mode ----------

def run_pairwise(args, prompts: list[dict], cand_client, judge_client, judge_rubric: str) -> dict:
    model_a, model_b = args.compare
    print(f"[generate] A={model_a}  B={model_b}  judge={args.judge}  n_prompts={len(prompts)}")
    t_start = time.time()

    def gen_both(p):
        out = {"tag": p["tag"]}
        for name, model in (("a", model_a), ("b", model_b)):
            try:
                out[name] = provider_chat(
                    cand_client, model, p["user"], system=args.system,
                    temperature=args.temperature, max_tokens=args.max_tokens,
                )
                out[f"{name}_error"] = None
            except Exception as e:
                out[name] = ""
                out[f"{name}_error"] = str(e)
        return out

    gens: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(gen_both, p) for p in prompts]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            gens[r["tag"]] = r
            ae = r.get("a_error")
            be = r.get("b_error")
            status = "ok" if not (ae or be) else f"A={ae or '-'} B={be or '-'}"
            print(f"  [{i:>2}/{len(prompts)}] gen {r['tag']:<20} {status}")

    import random
    rng = random.Random(args.seed)

    def judge(p):
        g = gens[p["tag"]]
        if g.get("a_error") or g.get("b_error"):
            return {"tag": p["tag"], "judgment": None, "error": "gen_failed"}
        flipped = rng.random() < 0.5
        if flipped:
            resp_a, resp_b = g["b"], g["a"]
        else:
            resp_a, resp_b = g["a"], g["b"]
        prompt_str = (
            f'PROMPT: {p["user"]}\n\n'
            f'RESPONSE_A: {resp_a}\n\n'
            f'RESPONSE_B: {resp_b}\n\n'
            'Compare on the rubric. Output JSON only.'
        )
        last_raw = ""
        last_err: str | None = None
        for attempt in range(2):
            try:
                raw = provider_chat(
                    judge_client, args.judge, prompt_str, system=judge_rubric,
                    temperature=0.0, max_tokens=600, think=False,
                )
                last_raw = raw
                parsed = parse_judge_json(raw)
                if parsed:
                    if flipped:
                        for k in ("winner", "voice_winner", "engagement_winner", "policy_winner"):
                            if parsed.get(k) == "A":
                                parsed[k] = "B"
                            elif parsed.get(k) == "B":
                                parsed[k] = "A"
                    return {"tag": p["tag"], "judgment": parsed, "raw": raw,
                            "flipped": flipped, "error": None}
                last_err = "parse_failed"
            except Exception as e:
                last_err = str(e)
        return {"tag": p["tag"], "judgment": None, "raw": last_raw,
                "flipped": flipped, "error": last_err}

    judgments: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(judge, p) for p in prompts]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            judgments[r["tag"]] = r
            status = "ok" if r.get("judgment") else f"FAIL({r.get('error')})"
            print(f"  [{i:>2}/{len(prompts)}] judge {r['tag']:<20} {status}")

    rows = []
    for p in prompts:
        g = gens.get(p["tag"], {})
        j = judgments.get(p["tag"], {})
        rows.append({
            "tag": p["tag"],
            "tier": p["tier"],
            "category": p["category"],
            "user": p["user"],
            "response_a": g.get("a", ""),
            "response_b": g.get("b", ""),
            "gen_a_error": g.get("a_error"),
            "gen_b_error": g.get("b_error"),
            "judgment": j.get("judgment"),
            "judge_error": j.get("error"),
        })

    return {
        "mode": "pairwise",
        "candidate_a": model_a,
        "candidate_b": model_b,
        "judge": args.judge,
        "system_prompt_used": args.system,
        "temperature": args.temperature,
        "n_prompts": len(prompts),
        "wall_seconds": round(time.time() - t_start, 1),
        "timestamp": datetime.now().isoformat(),
        "results": rows,
        "aggregates": _aggregate_pairwise(rows, model_a, model_b),
    }


def _aggregate_pairwise(rows: list[dict], model_a: str, model_b: str) -> dict:
    judged = [r for r in rows if r.get("judgment")]
    n = len(judged)
    if n == 0:
        return {"n_judged": 0}

    def win_rates(key):
        from collections import Counter
        c = Counter(r["judgment"].get(key, "tie") for r in judged)
        return {
            model_a: round(c.get("A", 0) / n, 3),
            model_b: round(c.get("B", 0) / n, 3),
            "tie":   round(c.get("tie", 0) / n, 3),
        }

    return {
        "n_judged": n,
        "n_total": len(rows),
        "n_gen_failed": sum(1 for r in rows if r.get("gen_a_error") or r.get("gen_b_error")),
        "win_rates_overall": win_rates("winner"),
        "win_rates_voice": win_rates("voice_winner"),
        "win_rates_engagement": win_rates("engagement_winner"),
        "win_rates_policy": win_rates("policy_winner"),
    }


# ---------- markdown report ----------

def write_markdown_absolute(result: dict, path: Path) -> None:
    a = result["aggregates"]
    lines = [
        f"# Eval: `{result['candidate']}` (absolute)",
        "",
        f"- judge: `{result['judge']}`  · prompts: {result['n_prompts']}  · wall: {result['wall_seconds']}s",
        f"- timestamp: {result['timestamp']}",
        "",
        "## Aggregates",
        "",
        f"- Voice fidelity (1-5): **{a.get('voice_fidelity_mean', '—')}**",
        f"- In-character rate: **{a.get('in_character_rate', '—')}**",
        f"- Expected-mode-match rate: **{a.get('expected_mode_match_rate', '—')}**",
    ]
    if "compound_noun_count_mean" in a:
        lines.append(f"- Compound-noun count (mean): **{a.get('compound_noun_count_mean', '—')}**")
    lines += [
        "",
        "### Voice fidelity by tier",
        "",
        "| tier | mean |",
        "|---|---|",
    ]
    for tier, v in (a.get("voice_fidelity_by_tier") or {}).items():
        lines.append(f"| {tier} | {v} |")
    lines += [
        "",
        "### Policy violations",
        "",
        "| flag | count / n_judged |",
        "|---|---|",
    ]
    for k, v in (a.get("policy_violations") or {}).items():
        lines.append(f"| {k} | {v} / {a['n_judged']} |")
    if "closer_type_distribution" in a:
        lines += [
            "",
            "### Closer-type distribution",
            "",
            "| closer | fraction |",
            "|---|---|",
        ]
        for k, v in (a.get("closer_type_distribution") or {}).items():
            lines.append(f"| {k} | {v} |")
    lines += [
        "",
        "## Per-prompt detail",
        "",
        "| tag | tier | voice | in-char | closer | mode-match | response (first 100 chars) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in result["results"]:
        j = r.get("judgment") or {}
        resp_preview = (r.get("response") or "")[:100].replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {r['tag']} | {r['tier']} | "
            f"{j.get('voice_fidelity', '—')} | "
            f"{j.get('in_character', '—')} | "
            f"{j.get('closer_type', '—')} | "
            f"{j.get('expected_mode_match', '—')} | "
            f"{resp_preview} |"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def write_markdown_pairwise(result: dict, path: Path) -> None:
    a = result["aggregates"]
    ma, mb = result["candidate_a"], result["candidate_b"]
    lines = [
        f"# Eval: `{ma}` vs `{mb}` (pairwise)",
        "",
        f"- judge: `{result['judge']}`  · prompts: {result['n_prompts']}  · wall: {result['wall_seconds']}s",
        f"- timestamp: {result['timestamp']}",
        "",
        "## Win rates",
        "",
        f"| dimension | {ma} | {mb} | tie |",
        f"|---|---|---|---|",
    ]
    for k, label in [
        ("win_rates_overall", "Overall"),
        ("win_rates_voice", "Voice"),
        ("win_rates_engagement", "Engagement"),
        ("win_rates_policy", "Policy"),
    ]:
        wr = a.get(k, {})
        lines.append(f"| {label} | {wr.get(ma, '—')} | {wr.get(mb, '—')} | {wr.get('tie', '—')} |")
    lines += [
        "",
        "## Per-prompt winners",
        "",
        "| tag | overall | voice | engagement | policy |",
        "|---|---|---|---|---|",
    ]
    for r in result["results"]:
        j = r.get("judgment") or {}
        lines.append(
            f"| {r['tag']} | {j.get('winner', '—')} | "
            f"{j.get('voice_winner', '—')} | "
            f"{j.get('engagement_winner', '—')} | "
            f"{j.get('policy_winner', '—')} |"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------- driver ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=None, help="project dir: defaults --model to its deploy.ollama_tag, --prompts to <project>/eval/prompts.jsonl, output to <project>/dataset/eval/, and loads eval/judge_rubric.md / eval/judge_pairwise.md if present (else falls back to built-in dec-bot defaults)")
    ap.add_argument("--model", help="model name (absolute mode); resolved via the configured provider")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"),
                    help="Compare two models (pairwise mode)")
    ap.add_argument("--judge", default="minimax-m2.7:cloud",
                    help="model name used as judge")
    ap.add_argument("--prompts", type=Path, default=None,
                    help="JSONL with eval prompts (default: <project>/eval/prompts.jsonl)")
    ap.add_argument("--system", default=None,
                    help="Override the candidate's system prompt (default: model default)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=500)
    ap.add_argument("--concurrency", type=int, default=3,
                    help="Parallel in-flight calls. 3 for cloud thinking-model judge; higher for local-only.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for pairwise A/B flip ordering")
    ap.add_argument("--tag", default=None,
                    help="Tag for the output filename (default: model name or A-vs-B)")
    args = ap.parse_args()
    events.set_stage("eval")

    proj = None
    judge_rubric_absolute = _DEC_BOT_ABSOLUTE
    judge_rubric_pairwise = _DEC_BOT_PAIRWISE
    cand_provider_spec = None
    judge_provider_spec = None

    if args.project:
        from pipeline.project import load_project
        proj = load_project(args.project)
        if not args.model and not args.compare:
            args.model = proj.deploy.ollama_tag or None
        if args.prompts is None:
            args.prompts = proj.p("eval/prompts.jsonl")
        out_dir = proj.dataset_path("eval")
        judge_rubric_absolute, judge_rubric_pairwise = _resolve_rubrics(proj)
        # Reuse the synthesis/triage provider configuration if available. Falls back to
        # whatever get_chat_provider() defaults to (Ollama Cloud) when None.
        cand_provider_spec = proj.synthesis.provider
        judge_provider_spec = proj.triage.provider or proj.synthesis.provider
    else:
        if args.prompts is None:
            args.prompts = DEFAULT_PROMPTS
        out_dir = DEFAULT_OUT_DIR

    if not args.model and not args.compare:
        ap.error("provide --model (absolute) or --compare A B (pairwise) — or --project with a deploy.ollama_tag set")
    if args.model and args.compare:
        ap.error("use either --model or --compare, not both")
    if not args.prompts.is_file():
        msg = f"no eval prompts at {args.prompts}"
        print(f"[eval] {msg}", file=sys.stderr)
        events.stage_end(status="error", exit_code=2, error=msg)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(args.prompts)
    print(f"[load] {len(prompts)} prompts from {args.prompts}")
    events.stage_start(command=[sys.executable, "-m", "pipeline.eval"] + sys.argv[1:],
                       params={"mode": "pairwise" if args.compare else "absolute",
                               "candidate": args.compare or args.model, "judge": args.judge,
                               "n_prompts": len(prompts), "concurrency": args.concurrency},
                       inputs=[str(args.prompts)], outputs=[str(out_dir)])
    started_at = time.monotonic()

    cand_client = get_chat_provider(cand_provider_spec)
    judge_client = get_chat_provider(judge_provider_spec)

    try:
        if args.compare:
            result = run_pairwise(args, prompts, cand_client, judge_client, judge_rubric_pairwise)
            tag = args.tag or f"{args.compare[0]}-vs-{args.compare[1]}".replace(":", "_").replace("/", "_")
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            json_path = out_dir / f"pairwise_{tag}_{ts}.json"
            md_path = out_dir / f"pairwise_{tag}_{ts}.md"
            json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
            write_markdown_pairwise(result, md_path)
        else:
            result = run_absolute(args, prompts, cand_client, judge_client, judge_rubric_absolute)
            tag = args.tag or args.model.replace(":", "_").replace("/", "_")
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            json_path = out_dir / f"absolute_{tag}_{ts}.json"
            md_path = out_dir / f"absolute_{tag}_{ts}.md"
            json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
            write_markdown_absolute(result, md_path)
    except Exception as e:
        events.stage_end(status="error", exit_code=1, duration_sec=time.monotonic() - started_at,
                         error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        raise

    print("\n" + "=" * 60)
    print(f"results → {json_path}")
    print(f"report  → {md_path}")
    print("=" * 60)
    print(json.dumps(result["aggregates"], indent=2))
    events.artifact(json_path, kind="stats")
    events.artifact(md_path, kind="stats")
    events.stage_end(status="ok", exit_code=0, duration_sec=time.monotonic() - started_at,
                     summary=result.get("aggregates"))


if __name__ == "__main__":
    main()
