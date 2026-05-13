"""
Ollama Cloud client wrapper.

Uses the ollama.com REST API. Requires OLLAMA_API_KEY environment variable.

Ollama Cloud API is OpenAI-compatible at the /v1 path and Ollama-native at /api.
We use /api/chat here for consistency with the Ollama Python SDK pattern.
"""

import os
import json
import time
import requests
from typing import Iterator

OLLAMA_HOST = "https://ollama.com"
DEFAULT_TIMEOUT = 600  # cloud generations can be long


class OllamaCloudError(Exception):
    pass


class OllamaCloudClient:
    def __init__(self, api_key: str | None = None, host: str = OLLAMA_HOST):
        self.api_key = api_key or os.environ.get("OLLAMA_API_KEY")
        if not self.api_key:
            raise OllamaCloudError(
                "OLLAMA_API_KEY not set. export OLLAMA_API_KEY=ollama_..."
            )
        self.host = host.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def chat(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        think: bool | None = None,
    ) -> str:
        """Non-streaming chat completion. Returns the assistant message content.

        When think is None: model default applies. When think is False: explicitly
        disables thinking-token generation (required for synthesis on kimi-k2.6,
        glm-4.7, deepseek-v4-flash, minimax-m2.* and other Ollama Cloud thinking
        models that otherwise return empty content while spending tokens on reasoning).
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
            },
        }
        if think is not None:
            payload["think"] = think

        last_err = None
        for attempt in range(max_retries):
            try:
                r = requests.post(
                    f"{self.host}/api/chat",
                    headers=self.headers,
                    json=payload,
                    timeout=DEFAULT_TIMEOUT,
                )
                if r.status_code == 429:
                    # rate-limited or queued; back off
                    wait = retry_delay * (2 ** attempt)
                    print(f"[rate-limited] sleeping {wait}s")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                data = r.json()
                return data["message"]["content"]
            except requests.RequestException as e:
                last_err = e
                wait = retry_delay * (2 ** attempt)
                print(f"[request failed: {e}] retrying in {wait}s")
                time.sleep(wait)

        raise OllamaCloudError(f"chat failed after {max_retries} retries: {last_err}")

    def chat_stream(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> Iterator[str]:
        """Streaming chat. Yields token strings."""
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
            },
        }
        r = requests.post(
            f"{self.host}/api/chat",
            headers=self.headers,
            json=payload,
            stream=True,
            timeout=DEFAULT_TIMEOUT,
        )
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "message" in chunk and "content" in chunk["message"]:
                yield chunk["message"]["content"]
            if chunk.get("done"):
                break
