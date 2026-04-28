import logging
from typing import List, Dict
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils import clean_and_parse_json

from ..base import BaseLLMModel

logger = logging.getLogger(__name__)

class GeminiModel(BaseLLMModel):
    """Google Gemini model using the modern google-genai SDK."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        **kwargs
    ):
        """
        Initialize Gemini model.
        
        Args:
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate (Max 8192)
            **kwargs: Additional Gemini client parameters
        """
        try:
            from google import genai
            from google.genai import types
            self.genai = genai
            self.types = types
        except ImportError:
            raise ImportError("New Google GenAI package not installed. Install with: pip install google-genai")
        
        self._model_name = model
        self.temperature = kwargs.get('temperature', 0)
        
        # Enforce the strict 8k output limit to prevent API 503/504 hangs
        self.max_tokens = min(kwargs.get('max_tokens', 4096), 8192)
        
        # Initialize the new Client architecture
        api_key = kwargs.get('api_key')
        if api_key:
            self.client = self.genai.Client(api_key=api_key)
        else:
            # Will automatically look for GEMINI_API_KEY environment variable
            self.client = self.genai.Client()
        
        logger.info(f"Initialized Gemini model: {model} using google-genai SDK")

    # Built-in robust retry logic to catch network hiccups and temporary 503s
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=5, min=10, max=120),
        before_sleep=lambda retry_state: logger.warning(f"⚠️ API Timeout or 503. Retrying in {retry_state.next_action.sleep} seconds...")
    )
    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """Chat with Gemini model using proper multi-turn conversation formatting."""
        logger.debug(f"Chatting with {self._model_name}")

        def _clip_text(text: str, max_chars: int = 4000) -> str:
            if len(text) <= max_chars:
                return text
            return f"{text[:max_chars]} ...[truncated {len(text) - max_chars} chars]"
        
        # Map generic messages to the strict google-genai Content schema
        formatted_contents = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Gemini expects the AI to be called 'model', not 'assistant'
            if role == "assistant":
                role = "model"
                
            formatted_contents.append(
                self.types.Content(
                    role=role,
                    parts=[self.types.Part.from_text(text=content)]
                )
            )

        # for different models, we need to set the thinking mode by different ways
        if self._model_name == "gemini-2.5-flash":
            config = self.types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                thinking_config=self.types.ThinkingConfig(
                    thinking_budget=0  # This explicitly disables thinking
                )
            )
        elif self._model_name == "gemini-2.5-pro":
            config = self.types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                thinking_config=self.types.ThinkingConfig(thinking_budget=128)
            )
        elif self._model_name == "gemini-3-flash-preview" or self._model_name == "gemini-3-flash" or self._model_name == "gemini-3-flash-lite" or self._model_name == "gemini-3.1-flash-lite-preview":
            config = self.types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                thinking_config=self.types.ThinkingConfig(thinking_level="low")
            )
        elif self._model_name == "gemini-3.1-pro-preview":
            config = self.types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                thinking_config=self.types.ThinkingConfig(thinking_level="low")
            )

        request_lines = [
            "=== GEMINI REQUEST ===",
            f"model={self._model_name}",
            f"temperature={self.temperature}",
            f"max_output_tokens={self.max_tokens}",
            f"message_count={len(messages)}",
            "messages:",
        ]
        for idx, msg in enumerate(messages, start=1):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            request_lines.append(
                f"[{idx}] role={role} chars={len(content)}\n{_clip_text(content)}"
            )
        request_lines.append("======================")
        logger.info("\n".join(request_lines))

        # Call the new endpoint
        response = self.client.models.generate_content(
            model=self._model_name,
            contents=formatted_contents,
            config=config
        )
        response_text = response.text or ""
        logger.info(
            "=== GEMINI RESPONSE ===\nmodel=%s\nresponse_chars=%d\n%s\n======================",
            self._model_name,
            len(response_text),
            _clip_text(response_text),
        )

        return response_text

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model_name