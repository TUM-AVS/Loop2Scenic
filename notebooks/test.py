# Add project root to Python path (must be before imports)
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Now import project modules
import logging
from src.embedding.embedder import Embedder
from src.config import get_config
import torch

logger = logging.getLogger(__name__)

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

inputs = [{
    "text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust.", 
    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
}]

embeddings = embedder.embed_documents(inputs)
print(f"Embeddings shape: {len(embeddings)} x {len(embeddings[0])}")