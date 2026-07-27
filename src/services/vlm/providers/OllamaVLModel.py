"""Ollama VLM via the OpenAI-compatible ``/v1`` chat API.

Reuses Qwen3Plus media handling (image/video → frame image_url parts) so the
critic / interpreter ``chat_with_content`` paths work unchanged.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

from src.services.ollama_utils import resolve_ollama_base_url
from src.services.vlm.providers.Qwen3Plus import Qwen3Plus

logger = logging.getLogger(__name__)

DEFAULT_VLM_MODEL = "qwen3-vl:32b"


class OllamaVLModel(Qwen3Plus):
    """Vision-language model served by Ollama (e.g. ``qwen3-vl:32b``)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        think: bool = True,
        **kwargs,
    ):
        if not api_key:
            from dotenv import load_dotenv

            load_dotenv()
            api_key = os.getenv("OLLAMA_API_KEY") or "ollama"

        resolved_base = resolve_ollama_base_url(base_url)
        resolved_model = model or model_name or DEFAULT_VLM_MODEL
        # Longer default: first Ollama load can be slow on large VLMs.
        kwargs.setdefault("timeout", 600)
        self.think = think

        super().__init__(
            api_key=api_key,
            base_url=resolved_base,
            model=resolved_model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        logger.info(
            "Initialized Ollama VLM: %s (endpoint=%s, temperature=%s, max_tokens=%s)",
            self._model_name,
            resolved_base,
            temperature,
            max_tokens,
        )

    @staticmethod
    def _message_text(message) -> str:
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return content.strip()
        for attr in ("reasoning", "reasoning_content", "thinking"):
            value = getattr(message, attr, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return (content or "").strip()

    def chat_with_content(
        self,
        contents: Any,
        system_instruction: Optional[str] = None,
        **kwargs,
    ) -> str:
        messages = self._to_openai_messages(contents, system_instruction)
        messages = self._demote_video_modality(messages)
        temperature = kwargs.pop("temperature", self.temperature)
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)
        think = kwargs.pop("think", self.think)
        extra_body = dict(kwargs.pop("extra_body", None) or {})
        if "think" not in extra_body:
            extra_body["think"] = think

        call_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ("temperature", "max_tokens", "timeout", "response_format")
        }
        call_kwargs["extra_body"] = extra_body

        start = time.perf_counter()
        response = None
        last_exc: Exception | None = None
        for attempt in range(1, 4):
            try:
                response = self.client.chat.completions.create(
                    model=self._model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **call_kwargs,
                )
                break
            except Exception as exc:
                last_exc = exc
                kind = self._classify_api_error(exc)
                logger.warning(
                    "Ollama VLM attempt %d/3 failed (%s): %s",
                    attempt,
                    kind,
                    str(exc)[:200],
                )
                if kind == "too_short":
                    messages = self._demote_video_modality(messages)
                    continue
                if kind in {"too_large", "input_length"}:
                    messages = self._shrink_visual_payload(messages, aggressiveness=attempt)
                    continue
                raise
        if response is None:
            raise last_exc  # type: ignore[misc]

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

        text = self._message_text(response.choices[0].message)
        if not text:
            raise ValueError(f"Empty response from Ollama VLM ({self._model_name})")
        return text
