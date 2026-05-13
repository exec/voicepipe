"""
Provider abstraction (minimal).

A *chat provider* turns (model, messages, options) into a string. A *embed provider* turns
(texts, model) into vectors. The pipeline's synthesis/triage stages currently use the Ollama
Cloud client directly; this module is the seam for adding OpenAI / local-vLLM / LM-Studio /
local-Ollama backends without touching the stages. `get_chat_provider(spec)` returns one.

`spec` is a small dict (from a `[providers.*]` table in project.toml, or hand-built):
    {"kind": "ollama_cloud"}                                  # OLLAMA_API_KEY from env
    {"kind": "openai_compat", "base_url": "...", "api_key_env": "OPENAI_API_KEY"}
    {"kind": "openai_compat", "base_url": "http://localhost:11434/v1"}   # local Ollama, no key
"""

from typing import Protocol


class ChatProvider(Protocol):
    def chat(self, model: str, messages: list[dict], *, temperature: float = 0.8,
             top_p: float = 0.95, think: bool | None = None) -> str: ...


class EmbedProvider(Protocol):
    def embed(self, texts: list[str], model: str) -> list[list[float]]: ...


def get_chat_provider(spec: dict | None = None):
    spec = spec or {"kind": "ollama_cloud"}
    kind = spec.get("kind", "ollama_cloud")
    if kind == "ollama_cloud":
        from pipeline.providers.ollama_cloud import OllamaCloudClient
        return OllamaCloudClient(api_key=spec.get("api_key"))
    if kind == "openai_compat":
        from pipeline.providers.openai_compat import OpenAICompatClient
        return OpenAICompatClient(base_url=spec.get("base_url", "http://localhost:11434/v1"),
                                  api_key_env=spec.get("api_key_env"), api_key=spec.get("api_key"))
    raise ValueError(f"unknown provider kind {kind!r}")
