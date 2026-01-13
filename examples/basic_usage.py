"""
Basic usage example of the RAG pipeline.

This example demonstrates:
1. Initializing the pipeline
2. Ingesting documents with tags
3. Querying with tag-based filtering
4. Getting source information
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
    print("RAG Pipeline - Basic Usage Example")
    print("="*80 + "\n")
    
    # Initialize pipeline
    print("1. Initializing RAG Pipeline...")
    pipeline = RAGPipeline()
    print("✓ Pipeline initialized\n")
    
    # Example: Ingest some sample documents
    print("2. Ingesting sample documents...")
    
    # Create some sample documents
    sample_docs = [
        {
            "text": "Python is a high-level programming language. It is widely used for web development, data science, and machine learning.",
            "tags": ["programming", "python", "tutorial"]
        },
        {
            "text": "Machine learning is a subset of artificial intelligence. It involves training models on data to make predictions.",
            "tags": ["ai", "machine-learning", "data-science"]
        },
        {
            "text": "Web development with Python often uses frameworks like Django and Flask. These frameworks make it easy to build web applications.",
            "tags": ["programming", "python", "web-development"]
        }
    ]
    
    # Ingest the documents
    texts = [doc["text"] for doc in sample_docs]
    tags = [doc["tags"] for doc in sample_docs]
    
    chunks = pipeline.processor.process_texts(texts=texts, tags=tags)
    doc_ids = pipeline.vectorstore.add_documents(chunks)
    
    print(f"✓ Ingested {len(doc_ids)} document chunks\n")
    
    # Show stats
    stats = pipeline.get_stats()
    print(f"Vector Store Stats:")
    print(f"  - Total documents: {stats['document_count']}")
    print(f"  - Collection: {stats['collection_name']}\n")
    
    # Example 1: Basic query without filtering
    print("3. Example Query #1: Basic query (no filtering)")
    print("-" * 80)
    query1 = "What is machine learning?"
    print(f"Query: {query1}\n")
    
    response1 = pipeline.query(query1, return_sources=True)
    print(f"Response:\n{response1['response']}\n")
    print(f"Sources: {len(response1['sources'])} document(s)\n")
    
    # Example 2: Query with tag filtering
    print("4. Example Query #2: Query with tag filtering")
    print("-" * 80)
    query2 = "Tell me about Python programming"
    filter_tags = ["python"]
    print(f"Query: {query2}")
    print(f"Filter tags: {filter_tags}\n")
    
    response2 = pipeline.query(
        query2,
        filter_tags=filter_tags,
        return_sources=True
    )
    print(f"Response:\n{response2['response']}\n")
    print(f"Sources ({len(response2['sources'])} document(s)):")
    for i, source in enumerate(response2['sources'], 1):
        print(f"  {i}. Tags: {source.get('tags', [])}")
    print()
    
    # Example 3: Query with different tags
    print("5. Example Query #3: Query with specific tags")
    print("-" * 80)
    query3 = "Explain AI and ML concepts"
    filter_tags = ["ai", "machine-learning"]
    print(f"Query: {query3}")
    print(f"Filter tags: {filter_tags}\n")
    
    response3 = pipeline.query(
        query3,
        filter_tags=filter_tags,
        return_sources=True,
        top_k=3
    )
    print(f"Response:\n{response3['response']}\n")
    
    print("="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
