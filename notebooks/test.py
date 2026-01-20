import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Now import project modules (after path setup)
import logging  # noqa: E402
from src.embedding.embedder import Embedder  # noqa: E402
from src.config import get_config  # noqa: E402
from src.vectorstore.milvus_store import MilvusVectorStore  # noqa: E402

logger = logging.getLogger(__name__)


def load_scenario_data_from_raw(data_raw_path=None):
    """
    Automatically fetch folders in data/raw and create dictionary structures for each scenario.
    
    For each subfolder, expects 3 files:
    - code.scenic (first file)
    - description.txt (second file)
    - video.mp4 (third file)
    
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
        
        # Get absolute path of video.mp4
        video_absolute_path = str(video_path.resolve())
        
        # Get absolute path of the folder
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

inputs = load_scenario_data_from_raw()
print(len(inputs), "Inputs loaded successfully")

# Load configuration from config/config.yaml
config = get_config()  # Loads from config/config.yaml by default

# Initialize embedder with config
# For Qwen models, model_path is required
embedder = Embedder(
    provider=config.embedding.provider,
    model_name=config.embedding.model_name,
    model_path=getattr(config.embedding, 'model_path', None),  # Qwen requires this
    device=config.embedding.device,
    batch_size=config.embedding.batch_size
)

print("Embedder initialized successfully")

# inputs = [{
#     "text": "A woman playing with her dog on a beach at sunset.",
#     "instruction": "Retrieve images or text relevant to the user's query.",
# }, {
#     "text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust."
# }, {
#     "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
# }, {
#     "text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust.", 
#     "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
# }, {
#     "text": "A cute cat.",
#     "instruction": "Retrieve images or text relevant to the user's query.",
#     "image": "https://media.istockphoto.com/id/1443562748/de/foto/s%C3%BC%C3%9Fe-ingwerkatze.jpg?s=1024x1024&w=is&k=20&c=heo_zqB7Y1h0PCHdJIakdqddnHCX9JsbzZMs2skLt70="
# }]

# # 5 embeddings will take about 20 min

# Test put the embeddings into the vector store
# Build connection args from config
if config.vector_db.use_lite:
    connection_args = {"uri": config.vector_db.lite_db_path}
    print(f"Using Milvus Lite: {connection_args['uri']}")
else:
    connection_args = {
        "host": config.vector_db.host,
        "port": config.vector_db.port
    }
    print(f"Using Milvus Server: {connection_args['host']}:{connection_args['port']}")

vector_store = MilvusVectorStore(
    embedder=embedder, 
    collection_name="avs",
    connection_args=connection_args  # Pass connection args!
)
print("Collection stats before adding documents:", vector_store.get_collection_stats())

vector_store.add_documents(inputs)

# See the collection stats
print("Collection stats after adding documents:", vector_store.get_collection_stats())

# Test retrieve the embeddings from the vector store
# results = vector_store.similarity_search_with_score(query={"text":"Find me a .", "image":"D:/study/Thesis/3-Code/ads-mrag/data/raw/image2.png"}, k=1)
# print(results)

# drop the collection
# vector_store.reset_collection()
# print("Collection stats after resetting:", vector_store.get_collection_stats())