"""
Centralized VLM service for vision-language tasks.
"""

import logging
from typing import Optional, Dict

from .base import BaseVLMModel
# Added GeminiVisionModel to your imports
from .models import Qwen3VLModel, OpenAIVisionModel, GeminiVisionModel 

logger = logging.getLogger(__name__)


class VLMService:
    """
    Centralized VLM service that provides multimodal chat interface.
    
    This service handles VLM calls for processing images, videos, and text together.
    """
    
    def __init__(
        self,
        provider: str = "qwen3vl",
        model_path: Optional[str] = None,
        model: Optional[str] = None,
        device: str = "cuda",
        torch_dtype: str = "fp16",
        temperature: float = 0.7,
        max_tokens: int = 512,
        fps: float = 1.0,
        max_frames: int = 64,
        default_instruction: str = "You are a helpful AI assistant.",
        **kwargs
    ):
        """
        Initialize VLM service.
        
        Args:
            provider: Model provider ('qwen3vl', 'openai_vision', 'gemini')
            model_path: Path to local model (for Qwen3VL)
            model: Model name (for OpenAI Vision and Gemini)
            device: Device to run on ('cuda', 'cpu')
            torch_dtype: Data type ('fp16', 'fp32', 'bf16')
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            fps: Frames per second for video processing
            max_frames: Maximum frames to extract from video
            default_instruction: Default instruction text
            **kwargs: Additional model-specific arguments (e.g., api_key)
        """
        self.provider = provider.lower()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.fps = fps
        self.max_frames = max_frames
        self.default_instruction = default_instruction
        
        # Initialize the appropriate model
        if self.provider == "qwen3vl":
            if not model_path:
                raise ValueError("model_path is required for Qwen3VL provider")
            
            self.model: BaseVLMModel = Qwen3VLModel(
                model_path=model_path,
                device=device,
                torch_dtype=torch_dtype,
                max_length=kwargs.get('max_length', 8192),
                fps=fps,
                max_frames=max_frames,
                default_instruction=default_instruction,
                temperature=temperature,
                max_tokens=max_tokens,
                **{k: v for k, v in kwargs.items() if k != 'max_length'}
            )
            
        elif self.provider == "openai_vision":
            model = model or "gpt-4-vision-preview"
            self.model: BaseVLMModel = OpenAIVisionModel(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
        elif self.provider == "gemini":
            model = model or "gemini-1.5-pro"
            self.model: BaseVLMModel = GeminiVisionModel(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=kwargs.pop("api_key", None),
                **kwargs
            )
            
        else:
            raise ValueError(
                f"Unsupported provider: {provider}. "
                f"Supported: qwen3vl, openai_vision, gemini"
            )
        
        logger.info(f"VLMService initialized with {provider} model: {self.model.model_name}")
    
    def chat(
        self,
        text: Optional[str] = None,
        image: Optional[str] = None,
        video: Optional[str] = None,
        instruction: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Chat with the VLM using multimodal inputs.
        
        Args:
            text: Text input (description, question, etc.)
            image: Path to image file
            video: Path to video file
            instruction: Optional instruction text (overrides default_instruction)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response string
        """
        logger.debug(f"Chatting with {self.model.model_name}")
        
        # Use provided instruction, or fallback to the service default
        active_instruction = instruction or self.default_instruction
        
        response = self.model.chat(
            text=text,
            image=image,
            video=video,
            instruction=active_instruction,
            temperature=kwargs.get('temperature', self.temperature),
            max_tokens=kwargs.get('max_tokens', self.max_tokens),
            fps=kwargs.get('fps', self.fps),
            max_frames=kwargs.get('max_frames', self.max_frames),
            **{k: v for k, v in kwargs.items() if k not in ['temperature', 'max_tokens', 'fps', 'max_frames', 'instruction']}
        )
        logger.debug("Response received")
        return response

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self.model.model_name


def get_vlm_service(
    provider: str = "qwen3vl",
    model_path: Optional[str] = None,
    model: Optional[str] = None,
    device: str = "cuda",
    torch_dtype: str = "fp16",
    temperature: float = 0.7,
    max_tokens: int = 512,
    fps: float = 1.0,
    max_frames: int = 64,
    default_instruction: str = "You are a helpful AI assistant.",
    **kwargs
) -> VLMService:
    """
    Factory function to create a VLM service.
    
    Args:
        provider: Model provider ('qwen3vl', 'openai_vision', 'gemini')
        model_path: Path to local model (for Qwen3VL)
        model: Model name (for OpenAI Vision and Gemini)
        device: Device to run on
        torch_dtype: Data type
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        fps: Frames per second for video processing
        max_frames: Maximum frames to extract from video
        default_instruction: Default instruction text
        **kwargs: Additional arguments (like api_key)
        
    Returns:
        VLMService instance
    """
    return VLMService(
        provider=provider,
        model_path=model_path,
        model=model,
        device=device,
        torch_dtype=torch_dtype,
        temperature=temperature,
        max_tokens=max_tokens,
        fps=fps,
        max_frames=max_frames,
        default_instruction=default_instruction,
        **kwargs
    )