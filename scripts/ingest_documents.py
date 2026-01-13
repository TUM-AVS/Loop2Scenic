"""
Script to ingest documents into the vector store.

Usage:
    python scripts/ingest_documents.py --source <path> --tags <tag1,tag2>
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import RAGPipeline
from src.utils import setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="Ingest documents into the RAG pipeline"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to document or directory"
    )
    parser.add_argument(
        "--tags",
        type=str,
        help="Comma-separated list of tags (e.g., 'finance,report,2024')"
    )
    parser.add_argument(
        "--directory",
        action="store_true",
        help="Treat source as directory"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Recursively search directories (default: True)"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom config file"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level="INFO")
    
    # Parse tags
    tags = None
    if args.tags:
        tags = [tag.strip() for tag in args.tags.split(",")]
    
    # Initialize pipeline
    print("Initializing RAG Pipeline...")
    pipeline = RAGPipeline()
    
    # Ingest documents
    print(f"Ingesting documents from: {args.source}")
    if tags:
        print(f"Tags: {tags}")
    
    doc_ids = pipeline.ingest_documents(
        source=args.source,
        tags=tags,
        is_directory=args.directory,
        recursive=args.recursive
    )
    
    print(f"\n✓ Successfully ingested {len(doc_ids)} document chunks")
    
    # Show stats
    stats = pipeline.get_stats()
    print(f"\nVector Store Stats:")
    print(f"  - Total documents: {stats['document_count']}")
    print(f"  - Collection: {stats['collection_name']}")


if __name__ == "__main__":
    main()
