import logging
import time
from typing import Dict, List

from tenacity import retry, stop_after_attempt, wait_exponential

from ..base import BaseLLMModel

logger = logging.getLogger(__name__)


class AnthropicModel(BaseLLMModel):
    """Anthropic Claude model (e.g., Sonnet family)."""

    def __init__(
        self,
        model: str = "claude-3-7-sonnet-20250219",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ):
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("Anthropic package not installed. Install with: pip install anthropic") from exc

        timeout = kwargs.pop("timeout", 120)
        self._model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.client = anthropic.Anthropic(**kwargs)
        self._metrics = {
            "calls": 0,
            "response_time_ms": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        logger.info("Initialized Anthropic model: %s", model)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        before_sleep=lambda retry_state: logger.warning(
            "Anthropic call failed. Retrying in %s seconds...", retry_state.next_action.sleep
        ),
    )
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        system_messages = [m.get("content", "") for m in messages if m.get("role") == "system"]
        system_prompt = "\n".join(system_messages).strip() or None
        anthropic_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            if role == "system":
                continue
            if role == "assistant":
                role = "assistant"
            else:
                role = "user"
            anthropic_messages.append({"role": role, "content": msg.get("content", "")})

        if not anthropic_messages:
            anthropic_messages = [{"role": "user", "content": ""}]

        start = time.perf_counter()
        response = self.client.messages.create(
            model=self._model_name,
            system=system_prompt,
            messages=anthropic_messages,
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
        prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        total_tokens = prompt_tokens + completion_tokens
        self._metrics["calls"] += 1
        self._metrics["response_time_ms"] += elapsed_ms
        self._metrics["prompt_tokens"] += prompt_tokens
        self._metrics["completion_tokens"] += completion_tokens
        self._metrics["total_tokens"] += total_tokens

        text_parts: List[str] = []
        for block in getattr(response, "content", []) or []:
            if getattr(block, "type", None) == "text":
                text_parts.append(getattr(block, "text", "") or "")
        return "".join(text_parts)

    @property
    def model_name(self) -> str:
        return self._model_name

    def get_metrics_snapshot(self) -> Dict[str, float]:
        return dict(self._metrics)
