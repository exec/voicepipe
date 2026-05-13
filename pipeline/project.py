"""
Project configuration schema for the corpus-to-character pipeline.

A *project* is a directory containing:
  - project.toml          — the configuration (everything below; scalars + small lists inline,
                            long prose blocks referenced by filename)
  - corpus/               — the reference texts that define the target voice
  - <glossary file>       — optional stylistic glossary (recurring phrases/constructions)
  - seeds/seed_pairs.jsonl — hand-written (user, assistant) example pairs, optionally mode-tagged
  - (referenced prose files: synth preamble, content rules, variety menus, triage rubric,
     per-mode description files, deploy system message)
  - dataset/              — produced by the pipeline: raw/, dedup/, triage/, final/, adapter/

`load_project(dir)` reads project.toml, resolves referenced files relative to the project dir,
and fills every unspecified field with a sensible default. The GUI front-end (the eventual
product) writes this same TOML; the CLI (`python -m pipeline <stage> --project DIR`) consumes it.

Design intent: everything a human configured by hand for the dec-bot proof-of-concept is a field
here, with a default. Point the pipeline at a corpus, accept the defaults, and it runs.
"""

import sys
import typing
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, List, Optional, Union

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - 3.10 fallback
    import tomli as tomllib  # type: ignore


# ---------------------------------------------------------------------------
# Leaf specs
# ---------------------------------------------------------------------------

@dataclass
class ModeSpec:
    """A 'mode' is a register the synthesized responses can be in (e.g. EXPOSITION,
    AUTOBIOGRAPHICAL, HISTORICAL-INDICTMENT). Each batch is anchored to one mode."""
    name: str
    description: str = ""                       # the per-mode instruction block (may be loaded from a file)
    weight: float = 1.0                         # relative weight in random mode selection
    corpus_anchor: Optional[str] = None         # filename in corpus/ that is this mode's primary voice anchor
    styles: list[str] = field(default_factory=list)            # mode-coupled style directives
    category_weights: Optional[dict[str, float]] = None        # override the global category distribution in this mode
    context_files: list[str] = field(default_factory=list)     # extra reference files injected ONLY in this mode (e.g. a facts dossier)


@dataclass
class CategorySpec:
    """A category of user prompt (greeting, mundane question, advice request, ...).
    Categories can be hand-written or auto-proposed from the corpus (see pipeline.categorize)."""
    name: str
    weight: float = 1.0


@dataclass
class LengthProfile:
    name: str
    description: str
    weight: float = 1.0


@dataclass
class CloserPattern:
    """A phrase that the synthesis model tends to over-produce. Dedup caps the fraction of the
    dataset allowed to contain it."""
    name: str
    regex: str
    cap: float = 0.10        # max fraction of pairs allowed to match


# ---------------------------------------------------------------------------
# Stage configs
# ---------------------------------------------------------------------------

@dataclass
class SynthesisConfig:
    model: str = "mistral-large-3:675b-cloud"
    alt_model: Optional[str] = None              # fallback model if the primary refuses/stalls
    # which LLM backend the model name refers to. None / {"kind":"ollama_cloud"} = Ollama Cloud
    # (needs OLLAMA_API_KEY); {"kind":"openai_compat","base_url":"...","api_key_env":"..."} = any
    # OpenAI-compatible endpoint (OpenAI, a local Ollama at http://localhost:11434/v1, vLLM, ...).
    provider: Optional[dict] = None
    temperature: float = 0.9
    top_p: float = 0.95
    pairs_per_batch: int = 12
    concurrency: int = 3                         # batches in flight at once
    think: Optional[bool] = False                # False = disable thinking-token generation (needed for thinking models)
    target: int = 3000                           # target total pairs (re-runnable; counts existing batches)
    balance_modes: bool = True                   # round-robin through modes equally
    shared_styles: list[str] = field(default_factory=list)     # style directives valid in every mode


@dataclass
class DedupConfig:
    cosine_threshold: float = 0.85               # drop pairs with >= this cosine similarity to a kept pair
    embed_model: str = "nomic-embed-text"        # local Ollama embedding model
    embed_base_url: str = "http://localhost:11434"
    skip_embed: bool = False                     # hash-only dedup if no local embeddings available
    closer_patterns: list[CloserPattern] = field(default_factory=list)


@dataclass
class TriageConfig:
    model: str = "deepseek-v4-pro"               # an LLM judge (a thinking model works well here)
    provider: Optional[dict] = None              # same shape as SynthesisConfig.provider; None = Ollama Cloud
    batch_size: int = 30                         # pairs scored per call
    min_keep: int = 4                            # keep pairs scored >= this (1-5 scale)
    concurrency: int = 3
    rubric: str = ""                             # the scoring instructions + flag definitions (loaded from a file)
    critical_flags: list[str] = field(default_factory=list)            # exact flag names that disqualify a pair
    critical_flag_substrings: list[str] = field(default_factory=list)  # substring tokens — any flag containing one is critical


@dataclass
class AssembleConfig:
    val_fraction: float = 0.05
    salvage_paths: list[str] = field(default_factory=list)   # extra (already-curated) jsonl files to mix into the pool
    seed: int = 42


@dataclass
class TrainConfig:
    base_model: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"   # a pre-quantized 4-bit base for QLoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ])
    max_seq_len: int = 1024
    batch_size: int = 1                          # per-device train batch
    grad_accum: int = 8
    epochs: float = 3.0
    lr: float = 2e-4
    optim: str = "paged_adamw_8bit"
    seed: int = 42


@dataclass
class DeployConfig:
    ollama_from: str = "llama3.1:8b"             # the base model layer (FROM line in the generated Modelfile)
    ollama_tag: str = ""                         # e.g. "execxd/dec-bot:v3" — where `ollama create` writes
    system_message: str = ""                     # the deployed model's system prompt (loaded from a file)
    parameters: dict[str, Any] = field(default_factory=lambda: {
        "temperature": 0.85, "top_p": 0.9, "repeat_penalty": 1.08, "num_predict": 400,
    })
    stop: list[str] = field(default_factory=list)              # stop strings (e.g. ["<|eot_id|>", "<|end_of_text|>"])
    gguf_outtype: str = "f16"                                  # adapter GGUF dtype
    base_model_id_override: Optional[str] = None               # non-quantized HF mirror for `convert_lora_to_gguf.py` (BNB bases need this)
    llama_cpp_dir: Optional[str] = None                        # path to a llama.cpp checkout (for convert_lora_to_gguf.py)


# ---------------------------------------------------------------------------
# Top-level project
# ---------------------------------------------------------------------------

@dataclass
class Project:
    name: str
    description: str = ""
    root: Path = field(default_factory=Path)     # the project directory (set by load_project)

    # --- corpus & inputs ---
    corpus_dir: str = "corpus"                   # relative to root
    glossary_file: Optional[str] = None          # e.g. "corpus/00_glossary.md"
    seeds_file: Optional[str] = "seeds/seed_pairs.jsonl"
    dataset_dir: str = "dataset"

    # --- synthesis prompt assembly (the system message is: preamble + variety_menus + mode block + content_rules + glossary) ---
    synth_preamble: str = ""                     # "You are generating training data imitating <X>..." (loaded from a file)
    variety_menus: str = ""                      # opening/address/closing variety directives (loaded from a file)
    content_rules: str = ""                      # absolute content requirements — slur prohibition, no post-cutoff figures, etc. (loaded from a file)

    # --- taxonomy ---
    modes: list[ModeSpec] = field(default_factory=list)
    categories: list[CategorySpec] = field(default_factory=list)
    length_profiles: list[LengthProfile] = field(default_factory=lambda: [
        LengthProfile("short", "50-100 words per response", 0.30),
        LengthProfile("medium", "100-250 words per response", 0.70),
    ])

    # --- stages ---
    synthesis: SynthesisConfig = field(default_factory=SynthesisConfig)
    dedup: DedupConfig = field(default_factory=DedupConfig)
    triage: TriageConfig = field(default_factory=TriageConfig)
    assemble: AssembleConfig = field(default_factory=AssembleConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    deploy: DeployConfig = field(default_factory=DeployConfig)

    # --- path helpers ---
    def p(self, rel: str) -> Path:
        """Resolve a path relative to the project root."""
        if self.root == Path():
            raise ValueError("Project.root is unset; construct via load_project rather than Project(...) directly")
        return (self.root / rel).resolve()

    def corpus_path(self) -> Path:
        return self.p(self.corpus_dir)

    def dataset_path(self, *parts: str) -> Path:
        return self.p(self.dataset_dir).joinpath(*parts)

    def mode(self, name: str) -> ModeSpec:
        for m in self.modes:
            if m.name == name:
                return m
        raise KeyError(f"no mode named {name!r} in project {self.name!r}")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

# Fields whose string value is treated as a filename (relative to root) and replaced by the file's text.
_TEXT_REF_FIELDS = {
    "synth_preamble", "variety_menus", "content_rules",
}


def _maybe_load_text(root: Path, value: Any) -> str:
    """If `value` looks like a path to an existing file under `root`, return its contents.
    Otherwise return `value` unchanged (so inline text in the TOML still works)."""
    if not isinstance(value, str):
        return value
    if not (len(value) < 256 and "\n" not in value):
        return value
    candidate = (root / value).resolve()
    # Resolve both sides so '../' segments are normalized before the containment check —
    # otherwise a string like "../../etc/passwd" would silently exfiltrate to an LLM.
    if not candidate.is_relative_to(root):
        return value
    if candidate.is_file():
        return candidate.read_text(encoding="utf-8")
    return value


_HINTS_CACHE: dict[type, dict[str, Any]] = {}


def _hints(cls):
    h = _HINTS_CACHE.get(cls)
    if h is None:
        h = typing.get_type_hints(cls)
        _HINTS_CACHE[cls] = h
    return h


def _build_dataclass(cls, data: dict, root: Path):
    """Recursively construct a dataclass `cls` from a TOML dict, applying defaults for
    anything missing and resolving text-reference fields."""
    if not is_dataclass(cls):
        return data
    kwargs = {}
    hints = _hints(cls)
    field_map = {f.name: f for f in fields(cls)}
    for key, raw in (data or {}).items():
        if key not in field_map:
            continue  # ignore unknown keys (forward-compat)
        ftype = hints.get(key, field_map[key].type)
        # nested dataclass?
        if is_dataclass(ftype) and isinstance(raw, dict):
            kwargs[key] = _build_dataclass(ftype, raw, root)
        # list of dataclasses? (TOML array-of-tables)
        elif (getattr(ftype, "__origin__", None) is list
              and isinstance(raw, list)
              and ftype.__args__ and is_dataclass(ftype.__args__[0])):
            kwargs[key] = [_build_dataclass(ftype.__args__[0], item, root) for item in raw]
        else:
            kwargs[key] = raw
    # resolve text references — scoped to Project so nested dataclasses can't accidentally
    # collide with these names (e.g. a future field called `content_rules` elsewhere).
    if cls is Project:
        for name in _TEXT_REF_FIELDS:
            if name in kwargs:
                kwargs[name] = _maybe_load_text(root, kwargs[name])
    # mode/triage description files: a ModeSpec.description that names a file gets loaded;
    # a TriageConfig.rubric that names a file gets loaded; ditto DeployConfig.system_message.
    if cls is ModeSpec and "description" in kwargs:
        kwargs["description"] = _maybe_load_text(root, kwargs["description"])
    if cls is TriageConfig and "rubric" in kwargs:
        kwargs["rubric"] = _maybe_load_text(root, kwargs["rubric"])
    if cls is DeployConfig and "system_message" in kwargs:
        kwargs["system_message"] = _maybe_load_text(root, kwargs["system_message"])
    return cls(**kwargs)


def _validate(proj: Project, *, seeds_explicit: bool) -> None:
    """Sanity asserts after a project is loaded. Raises ValueError on configurations that
    can't possibly run; emits stderr warnings for soft issues (empty corpus, missing default
    seeds) so freshly-scaffolded templates still load."""
    if proj.modes and sum(m.weight for m in proj.modes) <= 0:
        raise ValueError("mode weights sum to <= 0; at least one mode must have a positive weight")
    if proj.categories and not any(c.weight > 0 for c in proj.categories):
        raise ValueError("all category weights are 0; at least one category must have a positive weight")
    cdir = proj.corpus_path()
    if not cdir.is_dir():
        print(f"[project] warning: corpus_dir {cdir} does not exist", file=sys.stderr)
    elif not any(fp.suffix.lower() in (".txt", ".md") for fp in cdir.iterdir() if fp.is_file()):
        print(f"[project] warning: corpus_dir {cdir} has no .txt/.md files", file=sys.stderr)
    if proj.seeds_file:
        sp = proj.p(proj.seeds_file)
        if not sp.is_file():
            if seeds_explicit:
                raise ValueError(f"seeds_file is set to {proj.seeds_file!r} but {sp} does not exist")
            print(f"[project] warning: seeds_file default {proj.seeds_file!r} not found at {sp}", file=sys.stderr)


def load_project(path: str | Path) -> Project:
    """Load a project from a directory containing project.toml (or from the toml file itself)."""
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        toml_path = path / "project.toml"
        root = path
    else:
        toml_path = path
        root = path.parent
    if not toml_path.is_file():
        raise FileNotFoundError(f"no project.toml at {toml_path}")
    data = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    seeds_explicit = isinstance(data, dict) and "seeds_file" in data

    proj = _build_dataclass(Project, {**data, "root": root}, root)

    _validate(proj, seeds_explicit=seeds_explicit)
    # Glossary text is read lazily by stages that need it (synthesis); we just keep the path.
    return proj


def glossary_text(proj: Project) -> str:
    if not proj.glossary_file:
        return ""
    p = proj.p(proj.glossary_file)
    return p.read_text(encoding="utf-8") if p.is_file() else ""


def load_seeds(proj: Project) -> list[dict]:
    """Each seed is {"messages": [...], "mode": "..."}. Untagged seeds get the first mode's name."""
    import json
    if not proj.seeds_file:
        return []
    p = proj.p(proj.seeds_file)
    if not p.is_file():
        return []
    default_mode = proj.modes[0].name if proj.modes else "DEFAULT"
    out = []
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"[seeds] skipping malformed line {i} in {p}: {e}", file=sys.stderr)
            continue
        if not isinstance(obj, dict) or not isinstance(obj.get("messages"), list):
            print(f"[seeds] skipping line {i} in {p}: not a {{messages: [...]}} object", file=sys.stderr)
            continue
        obj.setdefault("mode", default_mode)
        out.append(obj)
    return out


def load_corpus(proj: Project) -> dict[str, str]:
    """{filename: text} for every .txt and .md in the corpus dir (excluding the glossary)."""
    cdir = proj.corpus_path()
    out = {}
    if not cdir.is_dir():
        return out
    glossary_name = Path(proj.glossary_file).name if proj.glossary_file else None
    for fp in sorted(cdir.iterdir()):
        if fp.suffix.lower() in (".txt", ".md") and fp.name != glossary_name:
            out[fp.name] = fp.read_text(encoding="utf-8")
    return out
