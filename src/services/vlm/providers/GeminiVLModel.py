import logging
from typing import Optional
from pathlib import Path
import time

from ..base import BaseVLMModel

logger = logging.getLogger(__name__)

class GeminiVisionModel(BaseVLMModel):
    """Google Gemini Vision and Video model."""

    def __init__(
        self,
        model: str = "gemini-1.5-pro", # 1.5-pro is highly recommended for complex multimodal tasks
        temperature: float = 0.7,
        max_tokens: int = 512,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the Gemini VLM model.
        
        Args:
            model: Model name (e.g., gemini-1.5-pro, gemini-1.5-flash)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            api_key: Optional API key. If not provided, it will look for the GOOGLE_API_KEY env var.
            **kwargs: Additional configuration
        """
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Google Generative AI package not installed. Install with: pip install google-generativeai")
        
        self._model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if api_key:
            genai.configure(api_key=api_key)
            
        logger.info(f"Initialized Gemini model: {model}")

    def chat(
        self,
        text: Optional[str] = None,
        image: Optional[str] = None,
        video: Optional[str] = None,
        instruction: Optional[str] = None,
        **kwargs
    ) -> str:
        """Chat with Gemini model using multimodal inputs (Text, Image, and Video)."""
        import google.generativeai as genai
        from PIL import Image

        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        
        logger.debug(f"Chatting with {self._model_name}")

        # Setup generation configuration
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        # Initialize the specific model instance (allows dynamic system instructions)
        model_kwargs = {"model_name": self._model_name}
        if instruction:
            model_kwargs["system_instruction"] = instruction
            
        model = genai.GenerativeModel(**model_kwargs)

        # Build content list for the prompt
        contents = []

        # 1. Handle Image
        if image:
            image_path = Path(image)
            if image_path.exists():
                img = Image.open(image_path)
                contents.append(img)
            else:
                logger.warning(f"Image file not found: {image}")

        # 2. Handle Video
        uploaded_video = None
        if video:
            video_path = Path(video)
            if video_path.exists():
                logger.info(f"Uploading video for Gemini processing: {video}")
                uploaded_video = genai.upload_file(path=str(video_path))
                
                # Videos require processing time on Google's servers before they can be queried
                while uploaded_video.state.name == 'PROCESSING':
                    logger.debug("Waiting for video processing to complete...")
                    time.sleep(2)
                    uploaded_video = genai.get_file(uploaded_video.name)
                    
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
            response = model.generate_content(
                contents,
                generation_config=generation_config
            )
            result = response.text
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise
        finally:
            # Cleanup: Always delete the uploaded video file to save on File API storage quota
            if uploaded_video:
                try:
                    genai.delete_file(uploaded_video.name)
                    logger.debug(f"Cleaned up temporary video file: {uploaded_video.name}")
                except Exception as e:
                    logger.warning(f"Failed to delete temporary video file: {e}")

        return result

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model_name