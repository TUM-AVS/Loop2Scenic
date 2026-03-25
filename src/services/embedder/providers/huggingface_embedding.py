from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from src.services.embedder import BaseEmbeddingModel

class HuggingFaceEmbedding(BaseEmbeddingModel):
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        # SentenceTransformer wraps both the model and tokenizer into one easy interface
        self.model = SentenceTransformer(model_name, **kwargs)

    def encode(self, inputs: List[Dict[str, Any]]) -> List[List[float]]:
        # Extract the text from your input dictionaries. 
        texts_to_encode = [item.get("text", "") for item in inputs]
        
        # .encode() returns a numpy array, so we convert it to nested Python lists
        embeddings = self.model.encode(texts_to_encode)
        return embeddings.tolist()
    
    @property
    def dimension(self) -> int:
        # SentenceTransformer has a built-in method for this
        return self.model.get_sentence_embedding_dimension()