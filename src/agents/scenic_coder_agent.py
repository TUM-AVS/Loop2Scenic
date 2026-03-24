import logging

from .base_agent import BaseAgent
from src.services import BaseLLMModel, MilvusVectorStore, BaseEmbeddingModel
from src.prompt import load_prompt
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ScenicCoderAgent(BaseAgent):
    def __init__(self, llm_service: BaseLLMModel, vector_store: MilvusVectorStore, embedder: BaseEmbeddingModel):
        super().__init__()
        self.llm_service = llm_service
        self.vector_store = vector_store
        self.embedder = embedder
        self.prompt_template = load_prompt("adapt_code")

    def process(self, state: dict) -> dict:
        return state

    def adapt_code(self, original_scenic_code: str, evaluation_result: Dict[str, Any], aim_dsl: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt the original scenic code to the aim DSL.
        Take the original scenic code, the evaluation result, and the aim DSL as input.
        Generate the new scenic code recursively, in each generation we put the previously generated scenic code as input to ensure compatibility.
        """

        # 1. get the components need to be modified in the aim DSL
        components_to_modify = {}
        for key, value in evaluation_result.items():
            if not value:
                components_to_modify[key] = aim_dsl[key] # get key and value from aim DSL

        # 2. search the vector store for similar snippets
        snippets = {}
        for key, value in components_to_modify.items():
            sentence = f"{key}: {value}" # "key" also act as tag searching in vector store
            sentence_embedding = self.embedder.encode([sentence])
            similar_snippets = self.vector_store.similarity_search(sentence_embedding, k=3)
            snippets[key] = similar_snippets
        

        
        
        

        formatted_prompt = self.prompt_template.format(original_scenic_code=original_scenic_code, aim_dsl=aim_dsl)
        response = self.llm_service.chat(formatted_prompt)
        json_response = self._clean_and_parse_json(response)
        if json_response:
            return json_response
        else:
            logger.error("Failed to parse JSON")
            return None