"""
Main RAG pipeline orchestration.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from langchain.schema import Document

from .config import Config, get_config
from .ingestion import DocumentProcessor, DocumentLoader, DocumentChunker
from .embedding import get_embedder
from .vectorstore import ChromaVectorStore, MilvusVectorStore
from .retrieval import Retriever
from .generation import get_generator, PromptTemplate
from .utils import setup_logging

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Main RAG pipeline that orchestrates all components.
    
    This class provides a high-level interface for the entire RAG workflow:
    - Document ingestion and processing
    - Embedding generation
    - Vector storage with tag-based filtering
    - Document retrieval
    - Response generation
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize RAG pipeline.
        
        Args:
            config: Configuration object (loads default if None)
        """
        # Load configuration
        self.config = config or get_config()
        
        # Setup logging
        setup_logging(
            level=self.config.logging.level,
            log_file=self.config.logging.file,
            log_format=self.config.logging.format
        )
        
        logger.info("Initializing RAG Pipeline")
        
        # Initialize components
        self._initialize_components()
        
        logger.info("RAG Pipeline initialized successfully")

    def _initialize_components(self):
        """Initialize all pipeline components."""
        # Document processing
        self.loader = DocumentLoader(
            supported_formats=self.config.ingestion.supported_formats
        )
        
        self.chunker = DocumentChunker(
            strategy=self.config.chunking.strategy,
            chunk_size=self.config.chunking.chunk_size,
            chunk_overlap=self.config.chunking.chunk_overlap,
            separators=self.config.chunking.separators
        )
        
        self.processor = DocumentProcessor(
            loader=self.loader,
            chunker=self.chunker,
            max_workers=self.config.ingestion.max_workers
        )
        
        # Embedding
        embedder_kwargs = {}
        if hasattr(self.config.embedding, 'model_path') and self.config.embedding.model_path:
            embedder_kwargs['model_path'] = self.config.embedding.model_path
        
        self.embedder = get_embedder(
            provider=self.config.embedding.provider,
            model_name=self.config.embedding.model_name,
            device=self.config.embedding.device,
            batch_size=self.config.embedding.batch_size,
            **embedder_kwargs
        )
        
        # Vector store
        if self.config.vector_db.provider == "milvus":
            # Milvus configuration
            connection_args = {
                "host": getattr(self.config.vector_db, "host", "localhost"),
                "port": getattr(self.config.vector_db, "port", "19530")
            }
            
            index_params = {
                "metric_type": self.config.vector_db.distance_metric.upper(),
                "index_type": getattr(self.config.vector_db, "index_type", "IVF_FLAT"),
                "params": {"nlist": getattr(self.config.vector_db, "nlist", 1024)}
            }
            
            search_params = {
                "metric_type": self.config.vector_db.distance_metric.upper(),
                "params": {"nprobe": getattr(self.config.vector_db, "nprobe", 10)}
            }
            
            self.vectorstore = MilvusVectorStore(
                embedder=self.embedder,
                collection_name=self.config.vector_db.collection_name,
                connection_args=connection_args,
                index_params=index_params,
                search_params=search_params
            )
        else:
            # ChromaDB (default)
            self.vectorstore = ChromaVectorStore(
                embedder=self.embedder,
                persist_directory=self.config.vector_db.persist_directory,
                collection_name=self.config.vector_db.collection_name,
                distance_metric=self.config.vector_db.distance_metric
            )
        
        # Retrieval
        self.retriever = Retriever(
            vectorstore=self.vectorstore,
            top_k=self.config.retrieval.top_k,
            similarity_threshold=self.config.retrieval.similarity_threshold,
            enable_reranking=self.config.retrieval.enable_reranking
        )
        
        # Generation
        self.generator = get_generator(
            provider=self.config.llm.provider,
            model=self.config.llm.model,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
            streaming=self.config.llm.streaming
        )

    def ingest_documents(
        self,
        source: Union[str, Path],
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
        is_directory: bool = False,
        recursive: bool = True
    ) -> List[str]:
        """
        Ingest documents into the vector store.
        
        Args:
            source: File or directory path
            metadata: Additional metadata
            tags: List of tags for filtering
            is_directory: Whether source is a directory
            recursive: Whether to search subdirectories (for directories)
            
        Returns:
            List of document IDs
        """
        logger.info(f"Ingesting documents from: {source}")
        
        # Process documents
        if is_directory:
            chunks = self.processor.process_directory(
                directory_path=source,
                metadata=metadata,
                tags=tags,
                recursive=recursive
            )
        else:
            chunks = self.processor.process_file(
                file_path=source,
                metadata=metadata,
                tags=tags
            )
        
        if not chunks:
            logger.warning("No chunks to ingest")
            return []
        
        # Add to vector store
        doc_ids = self.vectorstore.add_documents(chunks)
        
        logger.info(f"Ingested {len(doc_ids)} document chunks")
        
        return doc_ids

    def query(
        self,
        query: str,
        filter_tags: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        return_sources: bool = False,
        custom_prompt: Optional[PromptTemplate] = None
    ) -> Union[str, Dict[str, Any]]:
        """
        Query the RAG pipeline.
        
        Args:
            query: Query string
            filter_tags: Optional tags to filter by
            top_k: Number of documents to retrieve
            return_sources: Whether to return source information
            custom_prompt: Custom prompt template
            
        Returns:
            Generated response or dictionary with response and metadata
        """
        logger.info(f"Processing query: '{query[:50]}...'")
        
        # Retrieve relevant documents
        context_docs = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            filter_tags=filter_tags,
            return_scores=False
        )
        
        if not context_docs:
            response = "I couldn't find any relevant information to answer your question."
            if return_sources:
                return {"response": response, "sources": [], "num_sources": 0}
            return response
        
        # Generate response
        if return_sources:
            result = self.generator.generate_with_metadata(
                query=query,
                context_documents=context_docs,
                prompt_template=custom_prompt
            )
            return result
        else:
            response = self.generator.generate(
                query=query,
                context_documents=context_docs,
                prompt_template=custom_prompt
            )
            return response

    def get_stats(self) -> Dict[str, Any]:
        """
        Get pipeline statistics.
        
        Returns:
            Dictionary with statistics
        """
        return self.vectorstore.get_collection_stats()

    def reset(self):
        """Reset the vector store (delete all documents)."""
        logger.warning("Resetting vector store")
        self.vectorstore.reset_collection()
        logger.info("Vector store reset complete")
