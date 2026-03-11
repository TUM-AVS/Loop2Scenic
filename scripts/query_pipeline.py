"""
Script to query the RAG pipeline.

Usage:
    python scripts/query_pipeline.py
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
    
    # Query, you can comment out the unused part
    query_dict = {
        "text": "What is the main finding of the scenario?",
        "image": "/home/dellpro2/chenli/ads-mrag/ads-mrag/data/processed/test_data/testimage.png",
        "video": "/home/dellpro2/chenli/ads-mrag/ads-mrag/data/processed/test_data/testvideo.mp4"
    }
    results = pipeline.query_without_reranking(query_dict)
    for doc, score in results:
        print(f"Match found in folder: {doc.id}")
        print(f"Metadata: {doc.metadata}")
        print(f"Score: {score}")


if __name__ == "__main__":
    main()
