"""
Example of ingesting documents from files.

This example demonstrates:
1. Loading documents from a directory
2. Adding custom metadata and tags
3. Querying the ingested documents
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import RAGPipeline
from src.utils import setup_logging


def main():
    # Setup logging
    setup_logging(level="INFO")
    
    print("="*80)
    print("RAG Pipeline - File Ingestion Example")
    print("="*80 + "\n")
    
    # Initialize pipeline
    print("Initializing RAG Pipeline...")
    pipeline = RAGPipeline()
    print("✓ Pipeline initialized\n")
    
    # Example 1: Ingest a single file
    print("Example 1: Ingest a single file")
    print("-" * 80)
    
    # NOTE: Replace with your actual file path
    file_path = "data/raw/sample_document.txt"
    
    if Path(file_path).exists():
        doc_ids = pipeline.ingest_documents(
            source=file_path,
            tags=["example", "single-file"],
            metadata={"category": "tutorial"}
        )
        print(f"✓ Ingested {len(doc_ids)} chunks from {file_path}\n")
    else:
        print(f"File not found: {file_path}")
        print("Create a sample file in data/raw/ to test this example\n")
    
    # Example 2: Ingest a directory
    print("Example 2: Ingest a directory")
    print("-" * 80)
    
    directory_path = "data/raw"
    
    if Path(directory_path).exists():
        doc_ids = pipeline.ingest_documents(
            source=directory_path,
            tags=["example", "directory"],
            is_directory=True,
            recursive=True
        )
        print(f"✓ Ingested {len(doc_ids)} chunks from {directory_path}\n")
    else:
        print(f"Directory not found: {directory_path}\n")
    
    # Show stats
    stats = pipeline.get_stats()
    print(f"Vector Store Stats:")
    print(f"  - Total documents: {stats['document_count']}")
    print(f"  - Collection: {stats['collection_name']}\n")
    
    # Query the ingested documents
    if stats['document_count'] > 0:
        print("Querying ingested documents...")
        print("-" * 80)
        
        query = "What information is available in the documents?"
        print(f"Query: {query}\n")
        
        response = pipeline.query(
            query=query,
            filter_tags=["example"],
            return_sources=True
        )
        
        print(f"Response:\n{response['response']}\n")
        print(f"Number of sources: {response['num_sources']}")
    
    print("="*80)
    print("Example completed!")
    print("="*80)


if __name__ == "__main__":
    main()
