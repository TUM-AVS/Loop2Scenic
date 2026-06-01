from dataclasses import dataclass
from typing import List, Optional

from .scenario_document import ScenarioDocument


@dataclass
class RetrievalResult:
    """Output of retriever.retrieve() with ranking scores for the top hit."""

    scenarios: List[ScenarioDocument]
    best_similarity_score: Optional[float] = None
    best_rerank_score: Optional[float] = None
