"""
A minimal OpenAI-compatible chat/embeddings client (no `openai` package needed — just `requests`).

Works against anything that speaks the OpenAI REST shape at `<base_url>/chat/completions` and
`<base_url>/embeddings`: OpenAI itself, a local Ollama (`http://localhost:11434/v1`), vLLM,
LM Studio, Together, etc. Used via `pipeline.providers.get_chat_provider({"kind": "openai_compat", ...})`.
"""

import os
import time

import requests

_TIMEOUT = 600


class OpenAICompatClient:
    def __init__(self, base_url: str = "http://localhost:11434/v1", *,
                 api_key_env: str | None = None, api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or (os.environ.get(api_key_env) if api_key_env else None)
        self.headers = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

    def chat(self, model: str, messages: list[dict], *, temperature: float = 0.8,
             top_p: float = 0.95, think: bool | None = None,
             max_retries: int = 3, retry_delay: float = 5.0) -> str:
        payload = {"model": model, "messages": messages, "temperature": temperature,
                   "top_p": top_p, "stream": False}
        last = None
        for attempt in range(max_retries):
            try:
                r = requests.post(f"{self.base_url}/chat/completions", headers=self.headers,
                                  json=payload, timeout=_TIMEOUT)
                if r.status_code == 429:
                    time.sleep(retry_delay * (2 ** attempt)); continue
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except requests.RequestException as e:
                last = e
                time.sleep(retry_delay * (2 ** attempt))
        raise RuntimeError(f"chat failed after {max_retries} retries: {last}")

    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        r = requests.post(f"{self.base_url}/embeddings", headers=self.headers,
                          json={"model": model, "input": texts}, timeout=_TIMEOUT)
        r.raise_for_status()
        return [row["embedding"] for row in r.json()["data"]]
