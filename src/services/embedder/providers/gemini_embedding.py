"""
Gemini multimodal embedding model using the google-genai SDK.
(Inline Bytes Version)
"""

from __future__ import annotations

import logging
import mimetypes
import os
from typing import Any, Dict, List, Optional

from ..base import BaseEmbeddingModel

logger = logging.getLogger(__name__)


class GeminiEmbedding(BaseEmbeddingModel):
    """Google Gemini embedding model for text, image, and video inputs using inline bytes."""

    DEFAULT_DIMENSION = 3072
    MODEL_DIMENSIONS = {
        "gemini-embedding-2": 3072,
        "gemini-embedding-001": 3072,
    }

    def __init__(
        self,
        model_name: str = "gemini-embedding-2",
        api_key: Optional[str] = None,
        output_dimensionality: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "Google GenAI package not installed. Install with: pip install google-genai"
            ) from exc

        self.genai = genai
        self.model_name = model_name
        self.output_dimensionality = output_dimensionality
        self._dimension = output_dimensionality or self.MODEL_DIMENSIONS.get(
            model_name,
            self.DEFAULT_DIMENSION,
        )

        resolved_api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if resolved_api_key:
            self.client = self.genai.Client(api_key=resolved_api_key)
        else:
            self.client = self.genai.Client()

        logger.info(
            "Initialized Gemini embedder (Bytes mode): model=%s, dimension=%s",
            self.model_name,
            self._dimension,
        )

    def encode(self, inputs: List[Dict[str, Any]]) -> List[List[float]]:
        """Encode multimodal inputs into embedding vectors."""
        if not inputs:
            return []

        embeddings: List[List[float]] = []
        for item in inputs:
            try:
                contents = self._build_contents(item)
                config = self._build_embed_config()

                result = self.client.models.embed_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )

                embedding_values = self._extract_embedding_values(result)
                if embedding_values is None:
                    raise RuntimeError(f"Gemini embedder returned no embedding for input: {item}")

                embeddings.append(embedding_values)
            except Exception as exc:
                logger.error("Error during embedding: %s", exc)
                raise

        return embeddings

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self._dimension

    def _build_embed_config(self) -> Optional[Any]:
        if self.output_dimensionality is None:
            return None
        return {"output_dimensionality": self.output_dimensionality}

    def _build_contents(self, item: Dict[str, Any]) -> List[Any]:
        from google.genai import types

        contents: List[Any] = []

        # Handle text
        text = item.get("text")
        if text:
            contents.append(str(text))

        # Handle media (image or video)
        media_path = item.get("video") or item.get("video_path") or item.get("image") or item.get("image_path")
        if media_path:
            media_path = str(media_path)
            if not os.path.exists(media_path):
                raise FileNotFoundError(f"Media file not found at path: {media_path}")

            # Guess the MIME type (e.g., 'video/mp4' or 'image/jpeg')
            mime_type, _ = mimetypes.guess_type(media_path)
            if not mime_type:
                # Fallback if mimetypes can't determine it
                mime_type = "video/mp4" if "video" in item else "image/jpeg"

            logger.info("Reading %s as bytes (MIME type: %s)", os.path.basename(media_path), mime_type)

            # Read the file as raw bytes
            with open(media_path, "rb") as f:
                media_bytes = f.read()

            # Append the bytes directly into the contents payload
            contents.append(
                types.Part.from_bytes(data=media_bytes, mime_type=mime_type)
            )

        if not contents:
            contents.append("")

        return contents

    def _extract_embedding_values(self, result: Any) -> Optional[List[float]]:
        embeddings = getattr(result, "embeddings", None)
        if not embeddings:
            return None

        first_embedding = embeddings[0]
        values = getattr(first_embedding, "values", None)
        if values is None:
            return None

        return list(values)