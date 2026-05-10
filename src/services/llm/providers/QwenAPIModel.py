import logging
import os
import time
from typing import Dict, List

from tenacity import retry, stop_after_attempt, wait_exponential

from ..base import BaseLLMModel

logger = logging.getLogger(__name__)


class QwenAPIModel(BaseLLMModel):
    """Qwen API model via OpenAI-compatible endpoint."""

    def __init__(
        self,
        model: str = "qwen-max",
        temperature: float = 0.7,
        max_tokens: int = 512,
        base_url: str = "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
        **kwargs,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("OpenAI package not installed. Install with: pip install openai") from exc

        timeout = kwargs.pop("timeout", 120)
        api_key = kwargs.pop("api_key", None)
        if not api_key:
            from dotenv import load_dotenv

            load_dotenv()
            api_key = os.getenv("QWEN_API_KEY")
        if api_key:
            kwargs["api_key"] = api_key

        self._model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.client = OpenAI(base_url=base_url, **kwargs)
        self._metrics = {
            "calls": 0,
            "response_time_ms": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        logger.info("Initialized Qwen API model: %s, temperature: %s, max_tokens: %s", model, temperature, max_tokens)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        before_sleep=lambda retry_state: logger.warning(
            "Qwen API call failed. Retrying in %s seconds...", retry_state.next_action.sleep
        ),
    )
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        response_format = kwargs.get("response_format")

        start = time.perf_counter()
        request_kwargs = {
            "model": self._model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "extra_body": {"enable_thinking": True},
            "timeout": kwargs.get("timeout", self.timeout),
            **{
                k: v
                for k, v in kwargs.items()
                if k not in ["temperature", "max_tokens", "timeout", "response_format"]
            },
        }
        if response_format is not None:
            request_kwargs["response_format"] = response_format

        response = self.client.chat.completions.create(**request_kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)
        self._metrics["calls"] += 1
        self._metrics["response_time_ms"] += elapsed_ms
        self._metrics["prompt_tokens"] += prompt_tokens
        self._metrics["completion_tokens"] += completion_tokens
        self._metrics["total_tokens"] += total_tokens

        choices = getattr(response, "choices", None) or []
        if not choices:
            raise ValueError("Qwen API response has no choices.")
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        return content or ""

    @property
    def model_name(self) -> str:
        return self._model_name

    def get_metrics_snapshot(self) -> Dict[str, float]:
        return dict(self._metrics)

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    if not os.getenv("QWEN_API_KEY"):
        raise SystemExit(
            "Set DEEPSEEK_API_KEY in .env or the environment "
            "(or OPENAI_API_KEY if the OpenAI client should supply the key)."
        )

    model = QwenAPIModel(model="qwen3.6-plus", max_tokens=8192,temperature=0)
    content = """
    You are an expert in generating Scenic 3.0 code for autonomous vehicle testing scenarios.

Your task is to generate a new Spatial Relation component that defines the spatial positioning and relationships between vehicles in the scenario. You prioritize sampling feasibility, syntactic correctness, and network simplicity.

Your task is to generate exactly one Spatial Relation component that defines:
- Lanes and lane sections used by the scenario
- Spawn points (e.g., egoSpawnPt, advSpawnPt)
- Trajectories (if required)
- Minimal spatial parameters necessary for the scenario

**Absolute Output Constraint (Must Never Be Violated)**
You must ONLY generate Scenic code belonging to the Spatial Relation component.

Your output must contain only possible among:
1. Lane / lane section selection
2. Intersection selection (if required)
3. Spawn points
4. Trajectories
5. Spatial helper variables strictly required by other components
No explanations, no comments, no markdown, no JSON.
Output raw Scenic code only.

**Internal Reasoning Directive (DO NOT OUTPUT):**

You must internally follow these steps, but do NOT output your reasoning:

1. Understand the user requirements and identify required spatial relations.
Identify only the spatial relations that are strictly required.
2. Inspect the already determined components and identify which elements are already defined.
If an element (e.g., egoSpawnPt, advSpawnPt, trajectory) already exists, you MUST NOT redefine it.
Only define what is missing and strictly required.

3. Study the reference components and imitate their structure, variable naming, and patterns as closely as possible.

4. Select the simplest valid spatial relations using built-in Scenic specifiers.

5. Apply the Spatial Relation Hierarchy and Syntax Application Rules strictly.

6. Avoid complex chains, loops, or indirect relative positioning unless absolutely necessary.

7. If a relation would require violating:
- Random Control Flow Rule
- Syntax Application Rules
- Spatial Relation Hierarchy
omit it without approximation.

8. Final Validation
Ensure:
- No forbidden attributes are accessed
- No random-dependent control flow
- No conflicting specifiers
- The network is minimal and stable



**Spatial Relation Hierarchy**: 
- NetworkElement: An abstract class. It contains the properties:
   1. orientation: Optional[VectorField] 
   2. polygon: Union[Polygon, MultiPolygon]
   3. network: Network = None  # Link to parent network.

- LinearElement: An abstract class, inherited from NetworkElement. It contains the properties:
   1. centerline: PolylineRegion
   2. leftEdge: PolylineRegion
   3. rightEdge: PolylineRegion

- Road: Inherited from LinearElement. It is a road consisting of one or more lanes. Lanes are grouped into 1 or 2 instances of `LaneGroup`: forwardLanes and backwardLanes. It contains properties: 
   1. lanes: Tuple[Lane]  # The order of the lanes is arbitrary. To access lanes in order according to their geometry, use `LaneGroup.lanes`.
   2. forwardLanes: Union[LaneGroup, None] #: Group of lanes aligned with the direction of the road, if any.
   3. backwardLanes: Union[LaneGroup, None] # Group of lanes going in the opposite direction, if any.

- Lane: Inherited from LinearElement. It is a lane of a road. It contains properties: 
   1. road: Road # The road this lane belongs to.
   2. group: LaneGroup # parent lane group
   3. sections: Tuple[LaneSection]  # A Lane is made of multiple LaneSection pieces. The sections are in order from start to end
   4. adjacentLanes: Tuple[Lane] = ()  # adjacent lanes of same type, if any
   5. maneuvers: Tuple[Maneuver] = () # possible maneuvers upon reaching the end of this lane

- LaneSection: Inherited from LinearElement. The "Neighbor Finder". It is a section of a lane. It contains properties: 
   1. lane: Parent Lane object.
   2. group: LaneGroup # Grandparent lane group
   3. road: Road  # Grand-grandparent road.
   4. isForward: bool # (Important) Whether this lane section has the same direction as its parent road.
   5. adjacentLanes: Tuple[LaneSection] = () # Adjacent lanes of the same type, if any
   6. _laneToLeft: Union[LaneSection, None] = None # (Important) Adjacent lane of same type to the left, if any
   7. _laneToRight: Union[LaneSection, None] = None # (Important) Adjacent lane of same type to the right, if any
   8. _fasterLane: Union[LaneSection, None] = None # (Important) Faster adjacent lane of same type, if any
   9. _slowerLane: Union[LaneSection, None] = None # (Important) Slower adjacent lane of same type, if any

- Intersection: Child of NetworkElement. An intersection where multiple roads meet. Note: Not a LinearElement, so it has no centerline. It contains properties:
   1. roads: Tuple[Road] # Roads connecting to this intersection, in some order, preserving adjacency
   2. incomingLanes: Tuple[Lane] # lanes entering the intersection
   3. outgoingLanes: Tuple[Lane] # lanes exiting the intersection
   4. maneuvers: Tuple[Maneuver] # all possible maneuvers through the intersection
   5. is3Way: bool # Whether this is a 3-way intersection
   6. is4Way: bool # Whether this is a 4-way intersection
   7. isSignalized: bool # Whether this is a signalized intersection

***Available <Specifier>***:
1. Specifier for position:
  at <vector>
  in <region>
  contained in <region>
  on (<region> | <Object> | <vector>)
  offset by <vector>
  offset along <direction> by <vector>
  beyond <vector> by (<vector> | <scalar>) [from (<vector> | <OrientedPoint>)]
  visible [from (<Point> | <OrientedPoint>)]
  not visible [from (<Point> | <OrientedPoint>)]
  (left | right) of (<vector> | <OrientedPoint> | <Object>) [by <scalar>]
  (ahead of | behind) (<vector> | <OrientedPoint> | <Object>) [by <scalar>]
  (above | below) (<vector> | <OrientedPoint> | <Object>) [by <scalar>]
  following <vectorField> [from <vector>] for <scalar>
2. Specifier for orientation:
  facing <orientation>
  facing (toward | away from) <vector>
  apparently facing <heading> [from <vector>]


***Syntax Application Rules***:
1. The Centerline Rule
Since both Road and Lane are LinearElements, they both have a centerline. 
```
# Loose placement: Somewhere on a Road
road = Uniform(*network.roads)
pt1 = new OrientedPoint in road.centerline 

# Precise placement: Specific Lane
lane = Uniform(*network.lanes)
pt2 = new OrientedPoint in lane.centerline
```
But intersection does not inherit from LinearElement, so it has no centerline. 

2. The Neighbor Finder Rule
To find left/right neighbors, you must drill down to the Section level, because roads split and merge.
```
# AVOID this (Lane objects are too long to have a constant neighbor):
# neighbor = lane._laneToLeft 
# DO this (LaneSection):
section = Uniform(*lane.sections) # Pick a slice
if section._laneToLeft is not None:
    neighbor = section._laneToLeft
```
3. "New" Rule 
When you define a new OrientedPoint, you must use the new OrientedPoint syntax. 
```
# AVOID this (Old syntax):
OP = OrientedPoint on ego_lane.centerline
or 
OP = ahead of egoSpawnPt by 30
# DO this (New syntax):
OP = new OrientedPoint on ego_lane.centerline
```

4. Strict Ban on Random Variable Comparisons in Filters
You MUST NOT evaluate, access properties of, or compare previously sampled random variables (e.g., egoInitLane) inside filter() functions, lambda expressions, or standard if conditions. Because Scenic uses delayed evaluation, random variables are not resolved at the time filter() is executed.

If you need to restrict a newly sampled object based on a previously sampled random object, you MUST sample broadly first, and then use a require statement to enforce the relationship.
```
# INVALID
advInitLane = Uniform(*filter(lambda l: l.road == egoManeuver.endLane.road, intersection.incomingLanes))
egoLaneSec._laneToLeft.isForward != egoLaneSec.isForward
# Valid
intersection = Uniform(*filter(lambda i: i.is4Way, network.intersections))
egoLaneSec = Uniform(*egoLane.sections)
leftSec = egoLaneSec._laneToLeft
if leftSec is not None:
    neighborSec = leftSec

# INVALID (Do not use random variables inside lambdas):
# egoInitLane is a random variable, so accessing .road inside the lambda will fail
adv1Maneuver = Uniform(*filter(lambda m: m.type is ManeuverType.STRAIGHT and m.startLane.road != egoInitLane.road, intersection.maneuvers))
# Valid
# Use require statements to handle relational constraints
# 1. Filter only by static, known properties (like ManeuverType)
adv1Maneuver = Uniform(*filter(lambda m: m.type is ManeuverType.STRAIGHT, intersection.maneuvers))
# 2. Use 'require' to enforce the relationship with the random variable
require adv1Maneuver.startLane.road != egoInitLane.road
```

5. The Property "position" can only specified once with the same priority 
For example, if you want to define a pedestrain position is ahead of egoSpawnPt and also right of egoSpawnPt, you are not allowed to do it by define the SpawnPt as right of egoSpawnPt and ahead of egoSpawnPt, because right of egoSpawnPt and ahead of egoSpawnPt cannot be specified at the same time. You could define the a IntSpawnPt as a new OrientedPoint following egoInitLane.orientation from egoSpawnPt, and then define the pedestrain position as right of the IntSpawnPt.
```
# wrong example:
pedSpawnPt = new OrientedPoint right of (ahead of egoSpawnPt by 25) by 5
# correct example:
IntSpawnPt = new OrientedPoint following egoInitLane.orientation from egoSpawnPt for 
25
pedSpawnPt = new OrientedPoint right of IntSpawnPt by 5

# wrong example:
egoSpawnPt = new OrientedPoint in egoSection.centerline, behind advSpawnPt by Range(5, 10)
# correct example:
egoSpawnPt = new OrientedPoint behind advSpawnPt by Range(5, 10)
```

6. Usage of facing specifier
facing should be used in the following cases:
```
facing <orientation>
facing (toward | away from) <vector>
apparently facing <heading> [from <vector>]
```
But noticed that the orientation property of the class Lane is a VectorField, so you cannot use it directly in the facing specifier. 
```
# Wrong example:
advSpawnPt = new OrientedPoint at advInitLane.centerline.end, facing advInitLane.orientation

# Correct example (OrientedPoint on centerline automatically has orientation):
advSpawnPt = new OrientedPoint on advInitLane.centerline

# Correct example (Using following for specific distance):
advSpawnPt = new OrientedPoint following roadDirection from somePt for 10
```

7. Finding Intersections
Intersections are found in `network.intersections`. Do not check `isIntersection` on `Road` or `Lane` objects.
```
# Correct way to find a 4-way intersection:
intersection = Uniform(*filter(lambda i: i.is4Way, network.intersections))

# Correct way to find any intersection:
intersection = Uniform(*network.intersections)
```

8. Trajectory Rule
Normally trajectory is defined as a list of lanes for a vehicle to follow.
```
# Correct example:
egoTrajectory = [egoInitLane, egoManeuver.connectingLane, egoManeuver.endLane]
advTrajectory = [advInitLane, advManeuver.connectingLane, advManeuver.endLane]
```

9. Random Control Flow Rule
This is a hard rule. Violating it will produce invalid Scenic code.
You must NEVER use if, filter, or conditional expressions whose condition depends on:
- a variable created by Uniform(...)
- any property of a random object
- any comparison involving a random object


**Illustrative Examples (Structural Reference Only)**
All identifiers below are placeholders and must NOT be reused unless explicitly defined.
Example 1 of spatial relation component:
intersection = Uniform(*filter(lambda i: i.is4Way and i.isSignalized, network.intersections))

egoManeuver = Uniform(*filter(lambda m: m.type is ManeuverType.STRAIGHT, intersection.maneuvers))
egoInitLane = egoManeuver.startLane
egoSpawnPt = new OrientedPoint in egoInitLane.centerline

advManeuver = Uniform(*egoManeuver.conflictingManeuvers)
advTrajectory = [advManeuver.startLane, advManeuver.connectingLane, advManeuver.endLane]
advInitLane = advManeuver.startLane
advSpawnPt = new OrientedPoint in advInitLane.centerline

egoDir = egoSpawnPt.heading
advDir = advSpawnPt.heading

Example 2 of spatial relation component:
initLane = Uniform(*network.lanes)
egoSpawnPt = new OrientedPoint in initLane.centerline

**Inputs**:
User Requirements:
{user_criteria}

Already Determined Components (context only, do not redefine):
{ready_components}

Reference Components:
{reference_components}

**Output Format**:
Just output the raw Scenic code. No JSON, no markdown code blocks, no descriptions, no comments. Do not add any conversational text, explanations, or validation checks after the code block.

    """
    messages = [{"role": "user", "content": content}]
    print(model.chat(messages))
    print(model.get_metrics_snapshot())