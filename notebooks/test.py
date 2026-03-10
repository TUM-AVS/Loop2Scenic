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
import csv
from datetime import datetime

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import logging  # noqa: E402
import torch  # noqa: E402
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
    scenarios_with_missing_files = []  # Track scenarios with missing files
    
    # Iterate through all subdirectories in data/raw
    for subfolder in data_raw_path.iterdir():
        if not subfolder.is_dir():
            continue
        for subsubfolder in subfolder.iterdir():
            if not subsubfolder.is_dir():
                continue
            
            # Check for required files
            code_scenic_path = subsubfolder / "code.scenic"
            description_path = subsubfolder / "description.txt"
            video_path = subsubfolder / "video.mp4"
            
            # Track missing files for this scenario
            missing_files = []
            if not code_scenic_path.exists():
                missing_files.append("code.scenic")
                logger.warning(f"Missing code.scenic in {subsubfolder}")
            if not description_path.exists():
                missing_files.append("description.txt")
                logger.warning(f"Missing description.txt in {subsubfolder}")
            if not video_path.exists():
                missing_files.append("video.mp4")
                logger.warning(f"Missing video.mp4 in {subsubfolder}")
            
            # If any files are missing, record it and skip processing
            if missing_files:
                scenarios_with_missing_files.append({
                    "folder": str(subsubfolder),
                    "missing_files": missing_files
                })
                continue
            
            # Read code.scenic content
            try:
                with open(code_scenic_path, 'r', encoding='utf-8') as f:
                    code_scenic_content = f.read().strip()
            except Exception as e:
                logger.error(f"Error reading code.scenic from {subsubfolder}: {e}")
                scenarios_with_missing_files.append({
                    "folder": str(subsubfolder),
                    "missing_files": ["code.scenic (read error)"]
                })
                continue
            
            # Read description.txt content
            try:
                with open(description_path, 'r', encoding='utf-8') as f:
                    description_content = f.read().strip()
            except Exception as e:
                logger.error(f"Error reading description.txt from {subsubfolder}: {e}")
                scenarios_with_missing_files.append({
                    "folder": str(subsubfolder),
                    "missing_files": ["description.txt (read error)"]
                })
                continue
            
            # Get absolute paths
            video_absolute_path = str(video_path.resolve())
            folder_absolute_path = subsubfolder.resolve().as_posix()
            
            # Create dictionary structure
            scenario_dict = {
                "text": f"Here is a autonomous driving scenario, the description is {description_content}, and related scenic code is {code_scenic_content}",
                "instruction": "Retrieve images or text or vedio relevant to the user's query",
                "video": video_absolute_path,
                "folder_path": folder_absolute_path
            }
            
            results.append(scenario_dict)
    
    # Save scenarios with missing files to CSV
    if scenarios_with_missing_files:
        # Create CSV filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = data_raw_path / f"missing_files_report_{timestamp}.csv"
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['scenario_folder', 'missing_files', 'reason', 'timestamp']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for scenario in scenarios_with_missing_files:
                    missing_files_str = ', '.join(scenario['missing_files'])
                    reason = f"Not processed - Missing required files: {missing_files_str}"
                    writer.writerow({
                        'scenario_folder': scenario['folder'],
                        'missing_files': missing_files_str,
                        'reason': reason,
                        'timestamp': datetime.now().isoformat()
                    })
            
            logger.info(f"Missing files report saved to: {csv_path}")
            print(f"\n{'='*80}")
            print(f"SUMMARY: {len(scenarios_with_missing_files)} scenario(s) have missing files:")
            print(f"{'='*80}")
            for i, scenario in enumerate(scenarios_with_missing_files, 1):
                print(f"\n{i}. {scenario['folder']}")
                print(f"   Missing files: {', '.join(scenario['missing_files'])}")
            print(f"\n{'='*80}")
            print(f"Total scenarios processed successfully: {len(results)}")
            print(f"Total scenarios with missing files: {len(scenarios_with_missing_files)}")
            print(f"Missing files report saved to: {csv_path}")
            print(f"{'='*80}\n")
        except Exception as e:
            logger.error(f"Error saving missing files report to CSV: {e}")
            print(f"\n{'='*80}")
            print(f"SUMMARY: {len(scenarios_with_missing_files)} scenario(s) have missing files:")
            print(f"{'='*80}")
            for i, scenario in enumerate(scenarios_with_missing_files, 1):
                print(f"\n{i}. {scenario['folder']}")
                print(f"   Missing files: {', '.join(scenario['missing_files'])}")
            print(f"\n{'='*80}")
            print(f"Total scenarios processed successfully: {len(results)}")
            print(f"Total scenarios with missing files: {len(scenarios_with_missing_files)}")
            print(f"Error saving CSV report: {e}")
            print(f"{'='*80}\n")
    else:
        print(f"\n✓ All scenarios have all required files. Processed {len(results)} scenario(s) successfully.\n")
    
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


def main(add_documents=False, clear_collection=False):
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
    logger.info(f"  Total documents: {stats_before.get('document_count', 0)}")
    

    # Clear collection
    if clear_collection:
        logger.info("Clearing collection...")
        vector_store.reset_collection()
        logger.info("Collection cleared successfully")
        logger.info("Collection stats after clearing:")
        stats_after = vector_store.get_collection_stats()
        logger.info(f"  Total documents: {stats_after.get('total_documents', 0)}")
        return
    
    # Add documents to vector store
    if add_documents:
        # Load scenario data
        logger.info("Loading scenario data...")
        documents = load_scenario_data_from_raw()
        logger.info(f"Loaded {len(documents)} scenarios")
        
        if not documents:
            logger.warning("No documents loaded. Exiting.")
            return
        
        # Add documents to vector store one at a time with error handling
        logger.info("Adding documents to vector store (one at a time)...")
        failed_documents = []
        successful_count = 0
        
        for idx, doc in enumerate(documents, 1):
            try:
                logger.info(f"Adding document {idx}/{len(documents)}: {doc.get('folder_path', 'Unknown')}")
                vector_store.add_documents([doc])
                successful_count += 1
                logger.info(f"Successfully added document {idx}")
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Failed to add document {idx}: {error_msg}")
                failed_documents.append({
                    'index': idx,
                    'folder_path': doc.get('folder_path', 'Unknown'),
                    'error': error_msg,
                    'timestamp': datetime.now().isoformat()
                })
        
        # Log failed documents to CSV file under data/raw
        if failed_documents:
            data_raw_path = project_root / "data" / "raw"
            data_raw_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            failed_log_path = data_raw_path / f"failed_documents_{timestamp}.csv"
            
            try:
                with open(failed_log_path, 'w', newline='', encoding='utf-8') as csvfile:
                    fieldnames = ['index', 'folder_path', 'error', 'timestamp']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    
                    writer.writeheader()
                    for failed_doc in failed_documents:
                        writer.writerow(failed_doc)
                
                logger.warning(f"Failed to add {len(failed_documents)} document(s). Log saved to: {failed_log_path}")
                print(f"\n{'='*80}")
                print(f"SUMMARY: {len(failed_documents)} document(s) failed to add:")
                print(f"{'='*80}")
                for failed_doc in failed_documents:
                    print(f"\n  Index {failed_doc['index']}: {failed_doc['folder_path']}")
                    print(f"    Error: {failed_doc['error']}")
                print(f"\n{'='*80}")
                print(f"Total documents processed successfully: {successful_count}")
                print(f"Total documents failed: {len(failed_documents)}")
                print(f"Failed documents log saved to: {failed_log_path}")
                print(f"{'='*80}\n")
            except Exception as e:
                logger.error(f"Error saving failed documents log to CSV: {e}")
                print(f"\n{'='*80}")
                print(f"SUMMARY: {len(failed_documents)} document(s) failed to add:")
                print(f"{'='*80}")
                for failed_doc in failed_documents:
                    print(f"\n  Index {failed_doc['index']}: {failed_doc['folder_path']}")
                    print(f"    Error: {failed_doc['error']}")
                print(f"\n{'='*80}")
                print(f"Total documents processed successfully: {successful_count}")
                print(f"Total documents failed: {len(failed_documents)}")
                print(f"Error saving CSV log: {e}")
                print(f"{'='*80}\n")
        else:
            logger.info(f"All {successful_count} documents added successfully")
            print(f"\n✓ All {successful_count} document(s) added successfully.\n")
        
        # Check collection stats after adding documents
        logger.info("Collection stats after adding documents:")
        stats_after = vector_store.get_collection_stats()
        logger.info(f"  Total documents: {stats_after.get('total_documents', 0)}")
        return
    else:
        logger.info("Skipping document indexing (add_documents=False)")
        logger.info("Using existing documents in vector store")
    
    # Example queries for testing
    example_queries = [
        # {"text": "Find me a video with yield action."},
        # Uncomment and modify paths as needed:
        # {"text": "Find me a video similar to this image.", "image": "/home/dellpro2/chenli/ads-mrag/ads-mrag/data/raw/testimage.png"},
        {"text": "Please check the ego car behavior in this video, and find me a video has the same ego car behavior.", "video": "/home/dellpro2/chenli/ads-mrag/ads-mrag/data/processed/test_data/testvideo.mp4"},
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

        # Move the embedder to CPU and clear cuda cache
        embedder.model.to('cpu') 
        torch.cuda.empty_cache()
        
        logger.info(f"Retrieved {len(results)} documents")
        for j, (doc, score) in enumerate(results, 1):
            logger.info(f"  Result {j}: Score={score:.4f}")
            if doc.metadata.get("folder_path"):
                logger.info(f"    Folder path: {doc.metadata['folder_path']}")
        
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
                if doc.metadata.get("folder_path"):
                    logger.info(f"    Folder path: {doc.metadata['folder_path']}")


if __name__ == "__main__":
    main()