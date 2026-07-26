"""Shared helpers for Ollama OpenAI-compatible clients."""

from __future__ import annotations

import os
from typing import Optional

DEFAULT_OLLAMA_HOST = "10.147.17.29:11434"


def resolve_ollama_base_url(base_url: Optional[str] = None) -> str:
    """Resolve OpenAI-compatible base URL for a remote/local Ollama server.

    Precedence: explicit ``base_url`` → ``OLLAMA_URL`` → ``OLLAMA_HOST`` → default host.
    Always returns a URL ending in ``/v1``.
    """
    if base_url:
        url = base_url.strip().rstrip("/")
    else:
        env_url = (os.getenv("OLLAMA_URL") or "").strip().rstrip("/")
        if env_url:
            url = env_url
        else:
            host = (os.getenv("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST).strip()
            host = host.removeprefix("http://").removeprefix("https://")
            url = f"http://{host}"
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return url
