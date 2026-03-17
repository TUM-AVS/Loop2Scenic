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
from src.schema import MultimodalQuery

def main(): 
    # Setup logging
    setup_logging(level="INFO")
    
    # Initialize pipeline
    print("Initializing RAG Pipeline...")
    pipeline = RAGPipeline(mode="query")
    
    # Query, you can comment out the unused part
    query_dict = {
        "text": "Can you find a similar scenario to the one in this video?",
        # "image_path": "/home/dellpro2/chenli/ads-mrag/ads-mrag/data/processed/test_data/testimage.png",
        "video_path": "/home/dellpro2/chenli/ads-mrag/ads-mrag/data/processed/test_data/testvideo.mp4"
    }
    interpreted_query_object = MultimodalQuery(**{
        "text": pipeline.interpret_multimodal_query(MultimodalQuery(**query_dict)),
        "image_path": query_dict.get("image_path", None),
        "video_path": query_dict.get("video_path", None),
    })
    query_object = MultimodalQuery(**query_dict)
    
    if interpreted_query_object is not None:
        base_scenario_id = pipeline.query_with_reranking(query_object, interpreted_query_object)
        print(f"Base scenario ID: {base_scenario_id}")
    else:
        print("Failed to interpret multimodal query")

if __name__ == "__main__":
    main()
