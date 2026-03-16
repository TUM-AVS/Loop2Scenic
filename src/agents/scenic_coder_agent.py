from .base_agent import BaseAgent

class ScenicCoderAgent(BaseAgent):
    def __init__(self, name: str):
        super().__init__(name)

    def process(self, state: dict) -> dict:
        return state

    def adapt_code(self, state: dict) -> dict:
        return state