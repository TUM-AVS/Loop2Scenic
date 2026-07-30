"""Ollama LLM via the OpenAI-compatible ``/v1`` chat API."""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from src.services.ollama_utils import resolve_ollama_base_url

from ..base import BaseLLMModel

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "qwen3.6:35b"


class OllamaModel(BaseLLMModel):
    """Text LLM served by Ollama (OpenAI-compatible endpoint)."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        think: bool = False,
        **kwargs,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai"
            ) from exc

        timeout = kwargs.pop("timeout", 300)
        kwargs.pop("model_path", None)
        kwargs.pop("device", None)
        kwargs.pop("streaming", None)
        kwargs.pop("default_instruction", None)

        if not api_key:
            from dotenv import load_dotenv

            load_dotenv()
            api_key = os.getenv("OLLAMA_API_KEY") or "ollama"

        self._base_url = resolve_ollama_base_url(base_url)
        self._model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.think = think
        self.client = OpenAI(base_url=self._base_url, api_key=api_key, timeout=timeout)
        self._metrics = {
            "calls": 0,
            "response_time_ms": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        logger.info(
            "Initialized Ollama LLM: %s (endpoint=%s, temperature=%s, max_tokens=%s)",
            self._model_name,
            self._base_url,
            temperature,
            max_tokens,
        )

    @staticmethod
    def _message_text(message) -> str:
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return content
        # Some thinking models only populate reasoning when the budget is tight.
        for attr in ("reasoning", "reasoning_content", "thinking"):
            value = getattr(message, attr, None)
            if isinstance(value, str) and value.strip():
                return value
        return content or ""

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        before_sleep=lambda retry_state: logger.warning(
            "Ollama LLM call failed. Retrying in %s seconds...",
            retry_state.next_action.sleep,
        ),
    )
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        temperature = kwargs.pop("temperature", self.temperature)
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)
        timeout = kwargs.pop("timeout", self.timeout)
        think = kwargs.pop("think", self.think)
        extra_body = dict(kwargs.pop("extra_body", None) or {})
        if "think" not in extra_body:
            extra_body["think"] = think

        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            extra_body=extra_body,
            **kwargs,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(
            getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0
        )
        self._metrics["calls"] += 1
        self._metrics["response_time_ms"] += elapsed_ms
        self._metrics["prompt_tokens"] += prompt_tokens
        self._metrics["completion_tokens"] += completion_tokens
        self._metrics["total_tokens"] += total_tokens

        choices = getattr(response, "choices", None) or []
        if not choices:
            raise ValueError("Ollama response has no choices.")
        return self._message_text(choices[0].message)

    @property
    def model_name(self) -> str:
        return self._model_name

    def get_metrics_snapshot(self) -> Dict[str, float]:
        return dict(self._metrics)
