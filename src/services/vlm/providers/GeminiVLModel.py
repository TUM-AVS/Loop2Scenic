import logging
from typing import Any, Optional
from pathlib import Path
import time
import google.generativeai as genai
from PIL import Image
import os

from ..base import BaseVLMModel

logger = logging.getLogger(__name__)

class GeminiVLModel(BaseVLMModel):
    """Google Gemini Vision and Video model."""

    def __init__(
        self,
        model: str = "gemini-1.5-pro", # 1.5-pro is highly recommended for complex multimodal tasks
        temperature: float = 0.7,
        max_tokens: int = 51200,
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
        self._model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        self.default_instruction = "You are a helpful AI assistant."
        
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
        logger.debug(f"Chatting with {self._model_name}")

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
                generation_config=self.generation_config
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

    def chat_with_content(
        self,
        contents: Any,
        system_instruction: Optional[str] = None,
    ) -> str:
        """Chat with Gemini model using formatted content inputs."""
        system_instruction = system_instruction or self.default_instruction
        model_kwargs = {"model_name": self._model_name}
        if system_instruction:
            model_kwargs["system_instruction"] = system_instruction

        model = genai.GenerativeModel(**model_kwargs)
        response = model.generate_content(contents, generation_config=self.generation_config)
        return response.text

    def load_media(self, file_path: str):
        """
        Uploads a local media file to the Gemini API and ensures it is 
        fully processed and ready to be used in a prompt.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Media file not found at path: {file_path}")

        print(f"Uploading {os.path.basename(file_path)} to Gemini...")
        
        # 1. Upload the file to Google's servers
        uploaded_file = genai.upload_file(path=file_path)
        
        # 2. Check if it's a video. If so, we MUST wait for it to process.
        if uploaded_file.mime_type.startswith("video/"):
            print("Video detected. Waiting for processing to complete...")
            
            # Poll the API every 2 seconds until the state changes
            while uploaded_file.state.name == "PROCESSING":
                print(".", end="", flush=True)
                time.sleep(2)
                # Fetch the updated file status from the API
                uploaded_file = genai.get_file(uploaded_file.name)
                
            print() # Add a newline after the loading dots
            
            # If it failed to process, stop the execution
            if uploaded_file.state.name == "FAILED":
                raise ValueError(f"Video processing failed for {file_path}")
                
        print(f"Successfully loaded and ready: {uploaded_file.name}")
        
        # 3. Return the actual File object, which can now be appended directly to your `contents` list!
        return uploaded_file

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model_name