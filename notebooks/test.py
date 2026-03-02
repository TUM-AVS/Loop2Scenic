"""
RAG Pipeline for Autonomous Driving Scenarios.

This script implements a complete RAG (Retrieval-Augmented Generation) pipeline
for autonomous driving scenarios, including:
1. Data loading from raw scenario files
2. Document embedding and vector store indexing
3. Similarity search and retrieval
4. Optional reranking of results
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import logging  # noqa: E402
from src.embedding.embedder import Embedder  # noqa: E402
from src.config import get_config  # noqa: E402
from src.vectorstore.milvus_store import MilvusVectorStore  # noqa: E402
from src.retrieval.models.qwen_vl_reranker import QwenVLReranker  # noqa: E402

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_scenario_data_from_raw(data_raw_path=None):
    """
    Load scenario data from raw data directory.
    
    For each subfolder, expects 3 files:
    - code.scenic (scenic code file)
    - description.txt (scenario description)
    - video.mp4 (scenario video)
    
    Args:
        data_raw_path: Path to data/raw directory. If None, uses project_root/data/raw
        
    Returns:
        List of dictionaries, each containing:
        {
            "text": "Here is a autonomous driving scenario, the description is <description.txt>, and related scenic code is <code.scenic>",
            "instruction": "Retrieve images or text or vedio relevant to the user's query",
            "video": "<absolute path of video.mp4>",
            "folder_path": "<absolute path of the folder>"
        }
    """
    if data_raw_path is None:
        data_raw_path = project_root / "data" / "raw"
    else:
        data_raw_path = Path(data_raw_path)
    
    if not data_raw_path.exists():
        logger.warning(f"Data raw path does not exist: {data_raw_path}")
        return []
    
    results = []
    
    # Iterate through all subdirectories in data/raw
    for subfolder in data_raw_path.iterdir():
        if not subfolder.is_dir():
            continue
        
        # Check for required files
        code_scenic_path = subfolder / "code.scenic"
        description_path = subfolder / "description.txt"
        video_path = subfolder / "video.mp4"
        
        # Skip if any required file is missing
        if not code_scenic_path.exists():
            logger.warning(f"Missing code.scenic in {subfolder}")
            continue
        if not description_path.exists():
            logger.warning(f"Missing description.txt in {subfolder}")
            continue
        if not video_path.exists():
            logger.warning(f"Missing video.mp4 in {subfolder}")
            continue
        
        # Read code.scenic content
        try:
            with open(code_scenic_path, 'r', encoding='utf-8') as f:
                code_scenic_content = f.read().strip()
        except Exception as e:
            logger.error(f"Error reading code.scenic from {subfolder}: {e}")
            continue
        
        # Read description.txt content
        try:
            with open(description_path, 'r', encoding='utf-8') as f:
                description_content = f.read().strip()
        except Exception as e:
            logger.error(f"Error reading description.txt from {subfolder}: {e}")
            continue
        
        # Get absolute paths
        video_absolute_path = str(video_path.resolve())
        folder_absolute_path = subfolder.resolve().as_posix()
        
        # Create dictionary structure
        scenario_dict = {
            "text": f"Here is a autonomous driving scenario, the description is {description_content}, and related scenic code is {code_scenic_content}",
            "instruction": "Retrieve images or text or vedio relevant to the user's query",
            "video": video_absolute_path,
            "folder_path": folder_absolute_path
        }
        
        results.append(scenario_dict)
    
    return results


def build_rag_pipeline(config, collection_name="avs"):
    """
    Build and initialize the RAG pipeline components.
    
    Args:
        config: Configuration object
        collection_name: Name of the Milvus collection
        
    Returns:
        Tuple of (embedder, vector_store, reranker)
    """
    # Initialize embedder
    logger.info("Initializing embedder...")
    embedder = Embedder(
        provider=config.embedding.provider,
        model_name=config.embedding.model_name,
        model_path=getattr(config.embedding, 'model_path', None),
        device=config.embedding.device,
        batch_size=config.embedding.batch_size
    )
    logger.info("Embedder initialized successfully")
    
    # Build connection args from config
    if config.vector_db.use_lite:
        connection_args = {"uri": config.vector_db.lite_db_path}
        logger.info(f"Using Milvus Lite: {connection_args['uri']}")
    else:
        connection_args = {
            "host": config.vector_db.host,
            "port": config.vector_db.port
        }
        logger.info(f"Using Milvus Server: {connection_args['host']}:{connection_args['port']}")
    
    # Initialize vector store
    logger.info("Initializing vector store...")
    vector_store = MilvusVectorStore(
        embedder=embedder,
        collection_name=collection_name,
        connection_args=connection_args
    )
    logger.info("Vector store initialized successfully")
    
    # Initialize reranker (optional)
    reranker = None
    if config.reranking.model_path:
        logger.info("Initializing reranker...")
        reranker = QwenVLReranker(
            model_path=config.reranking.model_path,
        )
        logger.info("Reranker initialized successfully")
    
    return embedder, vector_store, reranker


def main(add_documents=True):
    """
    Main RAG pipeline execution.
    
    Args:
        add_documents: If True, add documents to vector store. If False, skip indexing
                       and only perform queries on existing documents.
    """
    # Load configuration
    logger.info("Loading configuration...")
    config = get_config()
    
    # Build RAG pipeline
    embedder, vector_store, reranker = build_rag_pipeline(config)
    
    # Check collection stats before adding documents
    logger.info("Collection stats before adding documents:")
    stats_before = vector_store.get_collection_stats()
    logger.info(f"  Total documents: {stats_before.get('total_documents', 0)}")
    
    # Add documents to vector store if requested
    if add_documents:
        # Load scenario data
        logger.info("Loading scenario data...")
        documents = load_scenario_data_from_raw()
        logger.info(f"Loaded {len(documents)} scenarios")
        
        if not documents:
            logger.warning("No documents loaded. Exiting.")
            return
        
        # Add documents to vector store
        logger.info("Adding documents to vector store...")
        vector_store.add_documents(documents)
        logger.info("Documents added successfully")
        
        # Check collection stats after adding documents
        logger.info("Collection stats after adding documents:")
        stats_after = vector_store.get_collection_stats()
        logger.info(f"  Total documents: {stats_after.get('total_documents', 0)}")
    else:
        logger.info("Skipping document indexing (add_documents=False)")
        logger.info("Using existing documents in vector store")
    
    # Example queries for testing
    example_queries = [
        {"text": "Find me a video of a car turning left."},
        # Uncomment and modify paths as needed:
        # {"text": "Find me a video similar to this image.", "image": "path/to/testimage.png"},
        # {"text": "Find me a video similar to this video.", "video": "path/to/testvideo.mp4"},
    ]
    
    # Perform retrieval for each query
    for i, query in enumerate(example_queries, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Query {i}: {query}")
        logger.info(f"{'='*80}")
        
        # Retrieve documents
        logger.info("Retrieving documents...")
        results = vector_store.similarity_search_with_score(
            query=query,
            k=config.retrieval.top_k
        )
        
        logger.info(f"Retrieved {len(results)} documents")
        for j, (doc, score) in enumerate(results, 1):
            logger.info(f"  Result {j}: Score={score:.4f}")
            logger.info(f"    Text: {doc.page_content[:100]}...")
            if doc.metadata.get("video"):
                logger.info(f"    Video: {doc.metadata['video']}")
        
        # Rerank if reranker is available
        if reranker and results:
            logger.info("Reranking results...")
            reranked_results = reranker.rerank_with_scores(
                query=query,
                documents=[doc for doc, _ in results],
                top_k=config.retrieval.rerank_top_k if config.retrieval.enable_reranking else None
            )
            
            logger.info(f"Reranked {len(reranked_results)} documents")
            for j, (doc, score) in enumerate(reranked_results, 1):
                logger.info(f"  Reranked Result {j}: Score={score:.4f}")
                logger.info(f"    Text: {doc.page_content[:100]}...")
                if doc.metadata.get("video"):
                    logger.info(f"    Video: {doc.metadata['video']}")


if __name__ == "__main__":
    main()