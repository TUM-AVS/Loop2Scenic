"""
Script to ingest scenarios into the vector store.

Usage:
    python scripts/ingest_documents.py --source <directory_path>
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
        description="Ingest scenarios into the RAG pipeline"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to directory containing scenario folders"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom config file"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level="INFO")
    
    # Validate source is a directory
    source_path = Path(args.source)
    if not source_path.exists():
        print(f"Error: Path does not exist: {args.source}")
        sys.exit(1)
    
    if not source_path.is_dir():
        print(f"Error: Source must be a directory: {args.source}")
        sys.exit(1)
    
    # Initialize pipeline
    print("Initializing RAG Pipeline...")
    pipeline = RAGPipeline()
    
    # Ingest scenarios
    print(f"Ingesting scenarios from: {args.source}")
    print("Step 1: Interpreting scenarios with VLM...")
    
    scenarios_dicts = pipeline.ingest_scenarios(directory_path=args.source)
    
    print(f"\n✓ Successfully processed {len(scenarios_dicts)} scenarios")
    print("✓ Scenario descriptions saved to new_description.txt files")
    
    # Show stats
    stats = pipeline.get_stats()
    print("\nVector Store Stats:")
    print(f"  - Total documents: {stats.get('document_count', 0)}")
    print(f"  - Collection: {stats.get('collection_name', 'N/A')}")


if __name__ == "__main__":
    main()
