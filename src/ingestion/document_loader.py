"""
Document loading functionality.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
)
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Loads documents from various file formats."""

    LOADER_MAPPING = {
        '.txt': TextLoader,
        '.pdf': PyPDFLoader,
        '.docx': Docx2txtLoader,
        '.md': UnstructuredMarkdownLoader,
        '.html': UnstructuredHTMLLoader,
    }

    def __init__(self, supported_formats: Optional[List[str]] = None):
        """
        Initialize document loader.
        
        Args:
            supported_formats: List of supported file extensions
        """
        self.supported_formats = supported_formats or list(self.LOADER_MAPPING.keys())

    def load_document(
        self,
        file_path: Union[str, Path],
        metadata: Optional[Dict] = None
    ) -> List[Document]:
        """
        Load a single document.
        
        Args:
            file_path: Path to the document
            metadata: Additional metadata to attach to the document
            
        Returns:
            List of Document objects
            
        Raises:
            ValueError: If file format is not supported
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        extension = file_path.suffix.lower()
        
        if extension not in self.supported_formats:
            raise ValueError(
                f"Unsupported file format: {extension}. "
                f"Supported formats: {self.supported_formats}"
            )
        
        loader_class = self.LOADER_MAPPING.get(extension)
        if not loader_class:
            raise ValueError(f"No loader found for extension: {extension}")
        
        try:
            loader = loader_class(str(file_path))
            documents = loader.load()
            
            # Add custom metadata
            if metadata:
                for doc in documents:
                    doc.metadata.update(metadata)
            
            # Add file metadata
            for doc in documents:
                doc.metadata.update({
                    'source': str(file_path),
                    'file_name': file_path.name,
                    'file_type': extension,
                })
            
            logger.info(f"Loaded {len(documents)} document(s) from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            raise

    def load_directory(
        self,
        directory_path: Union[str, Path],
        metadata: Optional[Dict] = None,
        recursive: bool = True
    ) -> List[Document]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory_path: Path to the directory
            metadata: Additional metadata to attach to all documents
            recursive: Whether to search subdirectories
            
        Returns:
            List of Document objects
        """
        directory_path = Path(directory_path)
        
        if not directory_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory_path}")
        
        all_documents = []
        
        # Get all files with supported extensions
        pattern = "**/*" if recursive else "*"
        
        for file_path in directory_path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                try:
                    documents = self.load_document(file_path, metadata)
                    all_documents.extend(documents)
                except Exception as e:
                    logger.warning(f"Skipping {file_path}: {str(e)}")
                    continue
        
        logger.info(
            f"Loaded {len(all_documents)} document(s) from {directory_path} "
            f"({'recursive' if recursive else 'non-recursive'})"
        )
        
        return all_documents
