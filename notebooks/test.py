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

# Load configuration from config/config.yaml
config = get_config()  # Loads from config/config.yaml by default

# # Initialize embedder with config
# # For Qwen models, model_path is required
embedder = Embedder(
    provider=config.embedding.provider,
    model_name=config.embedding.model_name,
    model_path=getattr(config.embedding, 'model_path', None),  # Qwen requires this
    device=config.embedding.device,
    batch_size=config.embedding.batch_size
)

inputs = [{
    "text": "A woman playing with her dog on a beach at sunset.",
    "instruction": "Retrieve images or text relevant to the user's query.",
}, {
    "text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust."
}, {
    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
}, {
    "text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust.", 
    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
}, {
    "text": "A cute cat.",
    "instruction": "Retrieve images or text relevant to the user's query.",
    "image": "https://media.istockphoto.com/id/1443562748/de/foto/s%C3%BC%C3%9Fe-ingwerkatze.jpg?s=1024x1024&w=is&k=20&c=heo_zqB7Y1h0PCHdJIakdqddnHCX9JsbzZMs2skLt70="
}]

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
    collection_name="test",
    connection_args=connection_args  # Pass connection args!
)
print("Collection stats before adding documents:", vector_store.get_collection_stats())

# vector_store.add_documents(inputs)

# See the collection stats
print("Collection stats after adding documents:", vector_store.get_collection_stats())

# Test retrieve the embeddings from the vector store
results = vector_store.similarity_search_with_score(query={"text":"Find me a picture which has an animal."}, k=1)
print(results)

# drop the collection
# vector_store.reset_collection()