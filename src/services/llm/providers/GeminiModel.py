import logging
from typing import List, Dict
from tenacity import retry, stop_after_attempt, wait_exponential

from ..base import BaseLLMModel

logger = logging.getLogger(__name__)

class GeminiModel(BaseLLMModel):
    """Google Gemini model using the modern google-genai SDK."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.7,
        max_tokens: int = 8192,  # Fixed: Max output limit is 8k, not 512k.
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
        self.temperature = temperature
        
        # Enforce the strict 8k output limit to prevent API 503/504 hangs
        self.max_tokens = min(max_tokens, 8192)
        
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
        wait=wait_exponential(multiplier=2, min=2, max=30),
        before_sleep=lambda retry_state: logger.warning(f"⚠️ API Timeout or 503. Retrying in {retry_state.next_action.sleep} seconds...")
    )
    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """Chat with Gemini model using proper multi-turn conversation formatting."""
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        
        logger.debug(f"Chatting with {self._model_name}")
        
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
        
        # Configure generation parameters
        config = self.types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        
        # Call the new endpoint
        response = self.client.models.generate_content(
            model=self._model_name,
            contents=formatted_contents,
            config=config
        )
        
        return response.text

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model_name