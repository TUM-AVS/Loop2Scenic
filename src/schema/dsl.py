from typing import List, Optional, TypedDict

class DSL(TypedDict):
    scenario: Optional[str]
    ego: Optional[str]
    adversarials: Optional[List[str]]
    spatial_relation: Optional[str]
    requirement_and_restrictions: Optional[str]