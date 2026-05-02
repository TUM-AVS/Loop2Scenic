import logging
from typing import Any, Optional
from pathlib import Path
import time
from PIL import Image
import os
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils import clean_and_parse_json

from ..base import BaseVLMModel

logger = logging.getLogger(__name__)

class GeminiVLModel(BaseVLMModel):
    """Google Gemini Vision and Video model using the modern google-genai SDK."""

    def __init__(
        self,
        model: str = "gemini-1.5-pro",
        temperature: float = 0,
        max_tokens: int = 8192,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the Gemini VLM model.
        
        Args:
            model: Model name (e.g., gemini-1.5-pro, gemini-1.5-flash)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate (Max 8192)
            api_key: Optional API key.
            **kwargs: Additional configuration
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
        self.max_tokens = min(max_tokens, 8192)
        
        self.default_instruction = "You are a helpful AI assistant."
        
        # Initialize the new Client architecture
        if api_key:
            self.client = self.genai.Client(api_key=api_key)
        else:
            self.client = self.genai.Client()
        self._metrics = {
            "calls": 0,
            "response_time_ms": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
            
        logger.info(f"Initialized Gemini VLM model: {model} using google-genai SDK with temperature {temperature} and max tokens {max_tokens}")

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=5, min=10, max=120),
        before_sleep=lambda retry_state: logger.warning(f"⚠️ API Timeout or 503. Retrying in {retry_state.next_action.sleep} seconds...")
    )
    def chat(
        self,
        text: Optional[str] = None,
        image: Optional[str] = None,
        video: Optional[str] = None,
        instruction: Optional[str] = None,
        **kwargs
    ) -> str:
        """Chat with Gemini model using multimodal inputs (Text, Image, and Video)."""
        logger.debug(f"Chatting with {self._model_name}")

        # Configure generation parameters including dynamic system instructions
        sys_instruct = instruction or self.default_instruction
        config = self.types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            system_instruction=sys_instruct
        )

        contents = []

        # 1. Handle Image
        if image:
            image_path = Path(image)
            if image_path.exists():
                img = Image.open(image_path)
                contents.append(img)
            else:
                logger.warning(f"Image file not found: {image}")

        # 2. Handle Video via the new client.files API
        uploaded_video = None
        if video:
            video_path = Path(video)
            if video_path.exists():
                logger.info(f"Uploading video for Gemini processing: {video}")
                uploaded_video = self.client.files.upload(file=str(video_path))
                
                # Videos require processing time on Google's servers
                while uploaded_video.state.name == 'PROCESSING':
                    logger.debug("Waiting for video processing to complete...")
                    time.sleep(2)
                    uploaded_video = self.client.files.get(name=uploaded_video.name)
                    
                if uploaded_video.state.name == 'FAILED':
                    logger.error("Video processing failed on Gemini servers.")
                else:
                    contents.append(uploaded_video)
            else:
                logger.warning(f"Video file not found: {video}")

        # 3. Handle Text
        if text:
            contents.append(text)

        # Fallback if no content is provided at all
        if not contents:
            contents.append("Please describe what you see.")

        # Generate the response
        try:
            start = time.perf_counter()
            response = self.client.models.generate_content(
                model=self._model_name,
                contents=contents,
                config=config
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            usage = getattr(response, "usage_metadata", None)
            prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
            completion_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
            total_tokens = int(getattr(usage, "total_token_count", prompt_tokens + completion_tokens) or 0)
            self._metrics["calls"] += 1
            self._metrics["response_time_ms"] += elapsed_ms
            self._metrics["prompt_tokens"] += prompt_tokens
            self._metrics["completion_tokens"] += completion_tokens
            self._metrics["total_tokens"] += total_tokens
            result = clean_and_parse_json(response.text)
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise
        finally:
            # Cleanup: Always delete the uploaded video file to save on File API storage quota
            if uploaded_video:
                try:
                    self.client.files.delete(name=uploaded_video.name)
                    logger.debug(f"Cleaned up temporary video file: {uploaded_video.name}")
                except Exception as e:
                    logger.warning(f"Failed to delete temporary video file: {e}")

        return result

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=30)
    )
    def chat_with_content(
        self,
        contents: Any,
        system_instruction: Optional[str] = None,
    ) -> str:
        """Chat with Gemini model using formatted content inputs."""
        sys_instruct = system_instruction or self.default_instruction
        config = self.types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            system_instruction=sys_instruct,
            response_mime_type="application/json"
        )

        start = time.perf_counter()
        response = self.client.models.generate_content(
            model=self._model_name,
            contents=contents,
            config=config
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        completion_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
        total_tokens = int(getattr(usage, "total_token_count", prompt_tokens + completion_tokens) or 0)
        self._metrics["calls"] += 1
        self._metrics["response_time_ms"] += elapsed_ms
        self._metrics["prompt_tokens"] += prompt_tokens
        self._metrics["completion_tokens"] += completion_tokens
        self._metrics["total_tokens"] += total_tokens
        json = clean_and_parse_json(response.text)
        if json:
            return json
        else:
            raise ValueError(f"Failed to parse JSON from response: {response.text}")

    def load_media(self, file_path: str):
        """
        Uploads a local media file to the Gemini API and ensures it is 
        fully processed and ready to be used in a prompt.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Media file not found at path: {file_path}")

        print(f"Uploading {os.path.basename(file_path)} to Gemini...")
        
        # 1. Upload the file via the new client API
        # By default, the API auto-detects mime_type from the extension
        uploaded_file = self.client.files.upload(file=file_path)
        
        # 2. Check if it's a video. If so, we MUST wait for it to process.
        # The new SDK object usually returns mime_type, but let's be safe and check if state exists
        if hasattr(uploaded_file, 'state') and uploaded_file.state is not None:
            print("Media processing required. Waiting...")
            
            # Poll the API every 2 seconds until the state changes
            while uploaded_file.state.name == "PROCESSING":
                print(".", end="", flush=True)
                time.sleep(2)
                # Fetch the updated file status from the API
                uploaded_file = self.client.files.get(name=uploaded_file.name)
                
            print() # Add a newline after the loading dots
            
            # If it failed to process, stop the execution
            if uploaded_file.state.name == "FAILED":
                raise ValueError(f"Media processing failed for {file_path}")
                
        print(f"Successfully loaded and ready: {uploaded_file.name}")
        
        # 3. Return the actual File object to append to `contents`
        return uploaded_file

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model_name

    def get_metrics_snapshot(self) -> dict:
        return dict(self._metrics)