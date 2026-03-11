"""
Generation module for creating responses using LLMs.

This module uses the centralized LLM service from src.llm.
"""

from .base import BaseGenerator
from .generator import Generator, get_generator
from .prompts import PromptTemplate, DEFAULT_QA_PROMPT

__all__ = [
    "BaseGenerator",
    "Generator",
    "get_generator",
    "PromptTemplate",
    "DEFAULT_QA_PROMPT",
]
