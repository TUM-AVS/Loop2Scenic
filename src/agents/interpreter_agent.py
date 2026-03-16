from .base_agent import BaseAgent

class InterpreterAgent(BaseAgent):
    def __init__(self, name: str):
        super().__init__(name)

    def process(self, state: dict) -> dict:
        return state

    def embed_query(self, state: dict) -> dict:
        return state

    def generate_dsl(self, state: dict) -> dict:
        return state