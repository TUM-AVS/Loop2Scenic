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
    parser.add_argument(
        "--tags",
        type=str,
        help="Comma-separated list of tags to filter by"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of documents to retrieve (default: 5)"
    )
    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Show source information"
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
    filter_tags = None
    if args.tags:
        filter_tags = [tag.strip() for tag in args.tags.split(",")]
    
    # Initialize pipeline
    print("Initializing RAG Pipeline...")
    pipeline = RAGPipeline()
    
    # Query
    print(f"\nQuery: {args.query}")
    if filter_tags:
        print(f"Filtering by tags: {filter_tags}")
    print("\n" + "="*80 + "\n")
    
    result = pipeline.query(
        query=args.query,
        filter_tags=filter_tags,
        top_k=args.top_k,
        return_sources=args.show_sources
    )
    
    if args.show_sources:
        print("Response:")
        print(result["response"])
        print("\n" + "="*80)
        print(f"\nSources ({result['num_context_docs']} documents):")
        for i, source in enumerate(result["sources"], 1):
            print(f"  {i}. {source['source']}")
            if source.get('tags'):
                print(f"     Tags: {', '.join(source['tags'])}")
    else:
        print(result)


if __name__ == "__main__":
    main()
