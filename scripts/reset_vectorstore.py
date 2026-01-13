"""
Script to reset the vector store (delete all documents).

Usage:
    python scripts/reset_vectorstore.py --confirm
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
        description="Reset the vector store"
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        required=True,
        help="Confirm that you want to delete all documents"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level="INFO")
    
    # Initialize pipeline
    print("Initializing RAG Pipeline...")
    pipeline = RAGPipeline()
    
    # Get current stats
    stats = pipeline.get_stats()
    print(f"\nCurrent state:")
    print(f"  - Documents: {stats['document_count']}")
    print(f"  - Collection: {stats['collection_name']}")
    
    # Confirm
    print("\n⚠️  WARNING: This will delete all documents from the vector store!")
    confirmation = input("Type 'DELETE' to confirm: ")
    
    if confirmation != "DELETE":
        print("Aborted.")
        return
    
    # Reset
    pipeline.reset()
    print("\n✓ Vector store reset successfully")


if __name__ == "__main__":
    main()
