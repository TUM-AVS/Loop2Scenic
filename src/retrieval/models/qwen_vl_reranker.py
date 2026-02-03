"""
QwenVL reranker implementation for reordering retrieved documents.
"""

import logging
import torch
from typing import List, Optional, Any, Union
from pathlib import Path
import sys

from langchain_core.documents import Document

from ..reranker import BaseReranker
from .qwen3_vl_reranker import Qwen3VLReranker as Qwen3VLRerankerModel


logger = logging.getLogger(__name__)


class QwenVLReranker(BaseReranker):
    """
    Reranker for QwenVL models.
    
    Uses Qwen3-VL-Reranker to rerank documents based on multimodal queries.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        instruction: str = "Retrieval relevant image or text with user's query",
        **kwargs
    ):
        """
        Initialize QwenVL reranker.
        
        Args:
            model_path: Path to local Qwen3-VL-Reranker model directory (preferred for local models)
            model_name_or_path: HuggingFace model name or path (for backward compatibility)
                If model_path is provided, this is ignored
            instruction: Instruction text for the reranker
            fps: Frames per second for video processing
            device: Device to run the model on (default: auto-detect)
            torch_dtype: Data type for the model (default: auto)
            max_frames: Maximum number of frames for video processing
            **kwargs: Additional arguments passed to Qwen3VLReranker
        """
        if Qwen3VLRerankerModel is None:
            raise ImportError(
                "Qwen3VLReranker model class not found. "
                "Please ensure qwen3_vl_reranker.py is available or install the required package."
            )
        self.model_path = model_path
        self.model = Qwen3VLRerankerModel(
            model_name_or_path=self.model_path,
            **kwargs
        )
        print(f"QwenVLReranker initialized successfully with model: {self.model_path}")

    def _document_to_dict(self, doc: Document) -> dict[str, Any]:
        """
        Convert a Document object to the dictionary format expected by the reranker.
        
        Args:
            doc: Document object with page_content and metadata
            
        Returns:
            Dictionary with text, image, video fields
        """
        doc_dict: dict[str, Any] = {}
        
        # Extract text from page_content
        if doc.page_content:
            doc_dict["text"] = doc.page_content
        
        # Extract multimodal fields from metadata
        if doc.metadata:
            # Check for image
            if "image" in doc.metadata:
                doc_dict["image"] = doc.metadata["image"]
            
            # Check for video
            if "video" in doc.metadata:
                doc_dict["video"] = doc.metadata["video"]
            
            # If we have both text and image, include both
            if "text" in doc.metadata and "image" in doc.metadata:
                # Create combined entry
                doc_dict = {
                    "text": doc.metadata.get("text", doc.page_content),
                    "image": doc.metadata["image"]
                }
        
        # If no content found, use empty text
        if not doc_dict:
            doc_dict = {"text": doc.page_content or ""}
        
        return doc_dict

    def _query_to_dict(self, query: Union[str, dict[str, Any]]) -> dict[str, Any]:
        """
        Convert query to the format expected by the reranker.
        
        Args:
            query: Query string or multimodal query dictionary
            
        Returns:
            Dictionary with text, image, video fields
        """
        if isinstance(query, str):
            return {"text": query}
        return query

    def rerank(
        self,
        query: Union[str, dict[str, Any]],
        documents: List[Document],
        top_k: Optional[int] = None,
        **kwargs
    ) -> List[Document]:
        """
        Rerank a list of documents based on a query.
        
        Args:
            query: Query string or multimodal query dictionary
            documents: List of documents to rerank
            top_k: Optional limit on number of documents to return after reranking
            **kwargs: Additional arguments (e.g., instruction, fps, max_frames)
            
        Returns:
            List of reranked Document objects, ordered by relevance (most relevant first)
        """
        if not documents:
            return []
        
        # Get scores using rerank_with_scores
        scored_docs = self.rerank_with_scores(query, documents, top_k=top_k, **kwargs)
        
        # Return only documents (without scores)
        return [doc for doc, _ in scored_docs]

    def rerank_with_scores(
        self,
        query: Union[str, dict[str, Any]],
        documents: List[Document],
        top_k: Optional[int] = None,
        **kwargs
    ) -> List[tuple[Document, float]]:
        """
        Rerank documents and return them with relevance scores.
        
        Args:
            query: Query string or multimodal query dictionary
            documents: List of documents to rerank
            top_k: Optional limit on number of documents to return
            **kwargs: Additional arguments (e.g., instruction, fps, max_frames)
            
        Returns:
            List of (Document, score) tuples, ordered by score (highest first)
        """
        if not documents:
            return []
        
        # Get parameters from kwargs or use defaults
        instruction = kwargs.get("instruction", self.instruction)
        fps = kwargs.get("fps", self.fps)
        max_frames = kwargs.get("max_frames", self.max_frames)
        
        # Convert query to dictionary format
        query_dict = self._query_to_dict(query)
        
        # Convert documents to dictionary format
        doc_dicts = [self._document_to_dict(doc) for doc in documents]
        
        logger.debug(
            f"Reranking {len(documents)} documents with query type: "
            f"{'text' if isinstance(query, str) else 'multimodal'}"
        )
        
        # Prepare input for Qwen3VLReranker.process method
        inputs = {
            "instruction": instruction,
            "query": query_dict,
            "documents": doc_dicts,
            "fps": fps
        }
        
        if max_frames is not None:
            inputs["max_frames"] = max_frames
        
        # Get scores from the reranker model
        try:
            scores = self.model.process(inputs)
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            raise
        
        # Convert scores to list of floats if they're tensors
        scores_list = []
        for score in scores:
            if isinstance(score, torch.Tensor):
                scores_list.append(float(score.item()))
            else:
                scores_list.append(float(score))
        
        # Validate scores
        if not scores_list or len(scores_list) != len(documents):
            logger.warning(
                f"Score count mismatch: expected {len(documents)}, got {len(scores_list) if scores_list else 0}"
            )
            # Return documents with dummy scores if there's a mismatch
            return [(doc, 0.0) for doc in documents]
        
        # Create list of (document, score) tuples
        doc_score_pairs = list(zip(documents, scores_list))
        
        # Sort by score (highest first) - assuming higher scores are better
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Apply top_k limit if specified
        if top_k is not None:
            doc_score_pairs = doc_score_pairs[:top_k]
        
        logger.info(
            f"Reranked {len(doc_score_pairs)} documents "
            f"(score range: {min(scores_list):.4f} - {max(scores_list):.4f})"
        )
        
        return doc_score_pairs
