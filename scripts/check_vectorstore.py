"""
Script to reset the vector store (delete all documents).

Usage:
    python scripts/check_vectorstore.py
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
    
    # Initialize pipeline
    print("Initializing RAG Pipeline...")
    pipeline = RAGPipeline()
    
    # Get current stats
    pipeline.get_stats()


if __name__ == "__main__":
    main()
