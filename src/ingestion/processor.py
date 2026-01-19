"""
Document processing pipeline.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.documents import Document
from tqdm import tqdm

from .document_loader import DocumentLoader
from .chunker import DocumentChunker

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Orchestrates document loading and chunking."""

    def __init__(
        self,
        loader: Optional[DocumentLoader] = None,
        chunker: Optional[DocumentChunker] = None,
        max_workers: int = 4
    ):
        """
        Initialize document processor.
        
        Args:
            loader: DocumentLoader instance
            chunker: DocumentChunker instance
            max_workers: Maximum number of parallel workers
        """
        self.loader = loader or DocumentLoader()
        self.chunker = chunker or DocumentChunker()
        self.max_workers = max_workers

    def process_file(
        self,
        file_path: Union[str, Path],
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Process a single file: load and chunk.
        
        Args:
            file_path: Path to the file
            metadata: Additional metadata
            tags: List of tags for filtering
            
        Returns:
            List of processed Document chunks
        """
        # Load document
        documents = self.loader.load_document(file_path, metadata)
        
        # Add tags to metadata if provided
        if tags:
            for doc in documents:
                doc.metadata['tags'] = tags
        
        # Chunk documents
        chunks = self.chunker.chunk_documents(documents)
        
        return chunks

    def process_directory(
        self,
        directory_path: Union[str, Path],
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
        recursive: bool = True,
        batch_processing: bool = True
    ) -> List[Document]:
        """
        Process all files in a directory.
        
        Args:
            directory_path: Path to the directory
            metadata: Additional metadata
            tags: List of tags for filtering
            recursive: Whether to search subdirectories
            batch_processing: Whether to use parallel processing
            
        Returns:
            List of processed Document chunks
        """
        directory_path = Path(directory_path)
        
        # Get all supported files
        all_files = []
        pattern = "**/*" if recursive else "*"
        
        for file_path in directory_path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.loader.supported_formats:
                all_files.append(file_path)
        
        if not all_files:
            logger.warning(f"No supported files found in {directory_path}")
            return []
        
        logger.info(f"Processing {len(all_files)} files from {directory_path}")
        
        all_chunks = []
        
        if batch_processing and len(all_files) > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(
                        self.process_file,
                        file_path,
                        metadata,
                        tags
                    ): file_path
                    for file_path in all_files
                }
                
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Processing documents"
                ):
                    try:
                        chunks = future.result()
                        all_chunks.extend(chunks)
                    except Exception as e:
                        file_path = futures[future]
                        logger.error(f"Error processing {file_path}: {str(e)}")
        else:
            # Sequential processing
            for file_path in tqdm(all_files, desc="Processing documents"):
                try:
                    chunks = self.process_file(file_path, metadata, tags)
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
        
        logger.info(f"Processed {len(all_files)} files into {len(all_chunks)} chunks")
        
        return all_chunks

    def process_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
        tags: Optional[List[List[str]]] = None
    ) -> List[Document]:
        """
        Process a list of text strings.
        
        Args:
            texts: List of text strings
            metadatas: List of metadata dicts (one per text)
            tags: List of tag lists (one per text)
            
        Returns:
            List of processed Document chunks
        """
        if metadatas and len(metadatas) != len(texts):
            raise ValueError("Length of metadatas must match length of texts")
        
        if tags and len(tags) != len(texts):
            raise ValueError("Length of tags must match length of texts")
        
        all_chunks = []
        
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            text_tags = tags[i] if tags else []
            
            if text_tags:
                metadata['tags'] = text_tags
            
            chunks = self.chunker.chunk_text(text, metadata)
            all_chunks.extend(chunks)
        
        logger.info(f"Processed {len(texts)} texts into {len(all_chunks)} chunks")
        
        return all_chunks
