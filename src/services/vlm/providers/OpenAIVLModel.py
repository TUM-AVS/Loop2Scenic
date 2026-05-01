import logging
import time
from typing import Optional

from ..base import BaseVLMModel

logger = logging.getLogger(__name__)

class OpenAIVLModel(BaseVLMModel):
    """OpenAI GPT-4 Vision model."""

    def __init__(
        self,
        model: str = "gpt-4-vision-preview",
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs
    ):
        """
        Initialize OpenAI Vision model.
        
        Args:
            model: Model name (gpt-4-vision-preview, gpt-4o, etc.)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional OpenAI client parameters
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
        
        self._model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(**kwargs)
        self._metrics = {
            "calls": 0,
            "response_time_ms": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        
        logger.info(f"Initialized OpenAI Vision model: {model}")

    def chat(
        self,
        text: Optional[str] = None,
        image: Optional[str] = None,
        video: Optional[str] = None,
        instruction: Optional[str] = None,
        **kwargs
    ) -> str:
        """Chat with OpenAI Vision model."""
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        
        logger.debug(f"Chatting with {self._model_name}")
        
        # Build messages
        messages = []
        
        if instruction:
            messages.append({"role": "system", "content": instruction})
        
        # Build content for user message
        content = []
        
        if text:
            content.append({"type": "text", "text": text})
        
        if image:
            # Read image and encode to base64
            import base64
            from pathlib import Path
            
            image_path = Path(image)
            if image_path.exists():
                with open(image_path, "rb") as image_file:
                    image_data = base64.b64encode(image_file.read()).decode('utf-8')
                    image_ext = image_path.suffix.lower()
                    mime_type = f"image/{image_ext[1:]}" if image_ext else "image/jpeg"
                    
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_data}"
                        }
                    })
        
        # Note: OpenAI Vision API doesn't support video directly
        if video:
            logger.warning("OpenAI Vision API doesn't support video input. Skipping video.")
        
        if not content:
            content.append({"type": "text", "text": "Please describe what you see."})
        
        messages.append({"role": "user", "content": content})
        
        # Call API
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **{k: v for k, v in kwargs.items() if k not in ['temperature', 'max_tokens']}
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
        
        return response.choices[0].message.content

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model_name

    def get_metrics_snapshot(self):
        return dict(self._metrics)