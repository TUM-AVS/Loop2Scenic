from typing import TypedDict

class EvalResult(TypedDict):
    scenario: bool
    ego: bool
    adversarials: bool
    spatial_relation: bool
    requirement_and_restrictions: bool