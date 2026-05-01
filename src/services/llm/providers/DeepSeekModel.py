import logging
import time
from typing import Dict, List

from tenacity import retry, stop_after_attempt, wait_exponential

from ..base import BaseLLMModel

logger = logging.getLogger(__name__)


class DeepSeekModel(BaseLLMModel):
    """DeepSeek API model via OpenAI-compatible endpoint."""

    def __init__(
        self,
        model: str = "deepseek-chat",
        temperature: float = 0.7,
        max_tokens: int = 512,
        base_url: str = "https://api.deepseek.com/v1",
        **kwargs,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("OpenAI package not installed. Install with: pip install openai") from exc

        timeout = kwargs.pop("timeout", 120)
        self._model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.client = OpenAI(base_url=base_url, **kwargs)
        self._metrics = {
            "calls": 0,
            "response_time_ms": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        logger.info("Initialized DeepSeek model: %s", model)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        before_sleep=lambda retry_state: logger.warning(
            "DeepSeek call failed. Retrying in %s seconds...", retry_state.next_action.sleep
        ),
    )
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=kwargs.get("timeout", self.timeout),
            **{
                k: v
                for k, v in kwargs.items()
                if k not in ["temperature", "max_tokens", "timeout"]
            },
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)
        self._metrics["calls"] += 1
        self._metrics["response_time_ms"] += elapsed_ms
        self._metrics["prompt_tokens"] += prompt_tokens
        self._metrics["completion_tokens"] += completion_tokens
        self._metrics["total_tokens"] += total_tokens

        choices = getattr(response, "choices", None) or []
        if not choices:
            raise ValueError("DeepSeek response has no choices.")
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        return content or ""

    @property
    def model_name(self) -> str:
        return self._model_name

    def get_metrics_snapshot(self) -> Dict[str, float]:
        return dict(self._metrics)
