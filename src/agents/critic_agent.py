from .base_agent import BaseAgent

class CriticAgent(BaseAgent):
    def __init__(self, name: str):
        super().__init__(name)

    def process(self, state: dict) -> dict:
        return state

    def evaluate_with_vlm(self, state: dict) -> dict:
        return state