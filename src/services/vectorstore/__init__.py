"""
Vector store module for storing and retrieving embeddings.
"""

from .milvus_store import MilvusVectorStore

__all__ = ["MilvusVectorStore"]
