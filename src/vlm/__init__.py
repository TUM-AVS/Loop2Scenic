"""
Centralized VLM (Vision Language Model) service module.
"""

from .vlm_service import VLMService, get_vlm_service
from .base import BaseVLMModel

__all__ = ["VLMService", "get_vlm_service", "BaseVLMModel"]
