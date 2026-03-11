"""
Script to query the RAG pipeline.

Usage:
    python scripts/query_pipeline.py --query "Your question here" --tags tag1,tag2
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
        description="Query the RAG pipeline"
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query string"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level="INFO")
    
    # Initialize pipeline
    print("Initializing RAG Pipeline...")
    pipeline = RAGPipeline()
    
    # Query
    print(f"\nQuery: {args.query}")
    results = pipeline.query_without_reranking(args.query)
    for doc, score in results:
        print(f"Match found in folder: {doc.id}")
        print(f"Metadata: {doc.metadata}")
        print(f"Score: {score}")


if __name__ == "__main__":
    main()
