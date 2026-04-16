from typing import List, Optional, TypedDict

class RoadSideStructure(TypedDict):
    object: str
    position: str

class TemporaryModification(TypedDict):
    object: str
    position: str

class DSL(TypedDict):
    """
    DSL example:
    {   
        'adversarials': [
            'A car travels forward in an adjacent lane and then performs a turn across the ego vehicle\'s path.'
        ],
        'ego': 'A car travels forward along its lane.',
        'requirements_and_restrictions': 'The scenario terminates when the ego vehicle has traveled a '
                                        'certain distance from its starting point.',
        'scenario': 'Ego vehicle travels straight on a multi-lane road while an adversarial vehicle turns '
                    'across its path from an adjacent lane.',
        'spatial_relation': 'The ego vehicle and the adversarial vehicle are positioned in different lanes '
                            'of a multi-lane road heading in the same direction, approaching a side street '
                            'intersection.',
        'road_side_structures': [
            {"object": "kiosk", "position": "Located on the intersection, from the left side of the ego vehicle."}
            {"object": "bench", "position": "Located on the intersection, from the right side of the ego vehicle."}
        ],
        'temporary_modifications': [
            {"object": "traffic cone", "position": "Located on the intersection, right on the lane of the ego vehicle."}
            {"object": "accident warning sign", "position": "Located on the intersection, right on the lane of the ego vehicle."}
        ]
    }
    """
    scenario: Optional[str]
    ego: Optional[str]
    adversarials: Optional[List[str]]
    spatial_relation: Optional[str]
    requirements_and_restrictions: Optional[str]

    # layer 2 road side structures
    road_side_structures: Optional[List[RoadSideStructure]]

    # layer 3 temporary modifications
    temporary_modifications: Optional[List[TemporaryModification]]