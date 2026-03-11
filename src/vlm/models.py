"""
VLM model implementations.
"""

import logging
from typing import Optional
import torch
import logging
import time
from pathlib import Path
from typing import Optional
from abc import ABC, abstractmethod

from .base import BaseVLMModel

logger = logging.getLogger(__name__)


class Qwen3VLModel(BaseVLMModel):
    """Qwen3VL model for vision-language tasks."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        torch_dtype: str = "fp16",
        max_length: int = 8192,
        fps: float = 1.0,
        max_frames: int = 64,
        default_instruction: str = "You are a helpful AI assistant.",
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs
    ):
        """
        Initialize Qwen3VL model.
        
        Args:
            model_path: Path to Qwen3VL model directory
            device: Device to run on ('cuda', 'cpu', etc.)
            torch_dtype: Data type ('fp16', 'fp32', 'bf16')
            max_length: Maximum sequence length
            fps: Frames per second for video processing
            max_frames: Maximum frames to extract from video
            default_instruction: Default instruction text
            **kwargs: Additional model parameters
        """
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError:
            raise ImportError("Transformers package not installed. Install with: pip install transformers")
        
        self._model_path = model_path
        self.device = device
        self.max_length = max_length
        self.fps = fps
        self.max_frames = max_frames
        self.default_instruction = default_instruction
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Convert torch_dtype string to torch dtype
        dtype_map = {
            "fp16": torch.float16,
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
        }
        self.torch_dtype = dtype_map.get(torch_dtype.lower(), torch.float16)
        
        logger.info(f"Loading Qwen3VL model from {model_path}")
        
        # Load model and processor
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self.torch_dtype,
            device_map=device,
            **kwargs
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        
        self.model.eval()
        
        logger.info(f"Qwen3VL model loaded successfully on {device}")

    def chat(
        self,
        text: Optional[str] = None,
        image: Optional[str] = None,
        video: Optional[str] = None,
        instruction: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Chat with Qwen3VL model using multimodal inputs.
        
        Args:
            text: Text input
            image: Path to image file
            video: Path to video file
            instruction: Optional instruction text
            **kwargs: Additional generation parameters
        """
        import torch
        
        temperature = kwargs.get('temperature', 0.7)
        max_new_tokens = kwargs.get('max_tokens', 512)
        
        logger.debug(f"Chatting with Qwen3VL model")
        
        # Prepare conversation format
        instruction = instruction or self.default_instruction
        
        # Build content list
        content = []
        
        if text:
            content.append({"type": "text", "text": text})
        
        if image:
            content.append({"type": "image", "image": image})
        
        if video:
            content.append({"type": "video", "video": video})
        
        if not content:
            content.append({"type": "text", "text": "Please describe what you see."})
        
        # Format conversation
        messages = [
            {"role": "system", "content": [{"type": "text", "text": instruction}]},
            {"role": "user", "content": content}
        ]
        
        # Prepare inputs
        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Process vision inputs
        try:
            from qwen_vl_utils.vision_process import process_vision_info
            
            image_patch_size = 16
            images, videos, video_kwargs = process_vision_info(
                messages,
                image_patch_size=image_patch_size,
                return_video_kwargs=True,
                return_video_metadata=True
            )
            
            if videos is not None:
                videos, video_metadata = zip(*videos)
                videos = list(videos)
                video_metadata = list(video_metadata)
            else:
                video_metadata = None
            
            # Update video kwargs
            video_kwargs.update({
                'fps': kwargs.get('fps', self.fps),
                'max_frames': kwargs.get('max_frames', self.max_frames)
            })
            
        except Exception as e:
            logger.warning(f"Error processing vision info: {e}")
            images = None
            videos = None
            video_metadata = None
            video_kwargs = {'do_sample_frames': False}
        
        # Process inputs
        inputs = self.processor(
            text=text_prompt,
            images=images,
            videos=videos,
            video_metadata=video_metadata,
            padding=True,
            return_tensors="pt",
            **video_kwargs
        )
        
        inputs = inputs.to(self.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )
        
        # Decode response
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return response_text.strip()

    @property
    def model_name(self) -> str:
        """Get model name."""
        return f"Qwen3VL-{self._model_path}"


class OpenAIVisionModel(BaseVLMModel):
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
        response = self.client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **{k: v for k, v in kwargs.items() if k not in ['temperature', 'max_tokens']}
        )
        
        return response.choices[0].message.content

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model_name

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