import logging
from typing import Optional
import torch

from ..base import BaseVLMModel

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