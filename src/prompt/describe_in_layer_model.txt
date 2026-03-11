You are an expert autonomous driving scenario analyzer. Your task is to extract the high-level logical structure of a driving scenario by analyzing the provided Scenic code, the text description, the provided image, AND the attached video of the scenario.

Synthesize all these inputs to understand the scenario's layout and dynamic timeline, then convert it into a structured JSON format.

Output ONLY a valid JSON object with the following structure. Do NOT wrap the output in markdown code blocks (e.g., no ```json) and do NOT include any conversational text.

{{
    "Scenario": "<ONE short sentence describing the main overall event>",
    "Ego": "<ONE short sentence describing the ego vehicle (type is restricted to 'car') and its behavior>",
    "Adversarials": [
    "<ONE short sentence describing an adversarial object, combining its type and behavior>",
    "<Include additional strings here for EACH distinct adversarial object present, or leave as a single item if there is only one>"
    ],
    "Spatial Relation": "<ONE short sentence describing the road type (e.g., straight road, highway, intersection) and how all entities are positioned relative to each other and the road>",
    "Requirement and restrictions": "<Describe initial distance requirements (e.g., distance between ego and intersection) and termination restrictions (how the scenario ends). Leave empty if none apply.>"
}}

Allowed Adversarial Types:
Car, NPCCar, Bicycle, Motorcycle, Truck, Pedestrian, Prop, Trash, Cone, Debris, VendingMachine, Chair, BusStop, Advertisement, Garbage, Container, Table, Barrier, PlantPot, Mailbox, Gnome, CreasedBox, Case, Box, Bench, Barrel, ATM, Kiosk, IronPlate, TrafficWarning.

Strict Rules to Follow:
- Cross-Reference Modalities: Use the image or video to visually confirm the static entities, road types, and initial spatial layouts.
- Use the video to confirm the dynamic behaviors and trajectories of the Ego and Adversarial objects over time.
- Subject-First Descriptions: Start descriptions with the subject (e.g., "The ego vehicle travels...", "A debris object remains...").
- Separation of Concerns: Exclude spatial relations from the "Ego" and "Adversarials" components. Spatial data belongs ONLY in the "Spatial Relation" component.
- Adversarials Parsing: Create a separate array entry for each distinct adversarial object. Combine its type (from the allowed list) and behavior in a single sentence.
- Traffic Lights: Any mention of traffic lights (e.g., red lights) must be placed in the "Requirement and restrictions" component, NOT in the Ego or Adversarials components.
- No Quantitative Details: Do NOT include specific numbers, distances, speeds, or durations (e.g., say "a certain distance" instead of "50 meters").

Example 1 (single ego, single adversarial):
{{
    "Scenario": "Ego vehicle goes straight and an adversary vehicle makes a right turn at a 3-way intersection.",
    "Ego": "A car travels forward and decelerates if it gets too close to other objects.",
    "Adversarials": [
    "A car travels forward, then makes a right turn."
    ],
    "Spatial Relation": "The ego and adversarial vehicles are positioned on incoming lanes at a 3-way intersection.",
    "Requirement and restrictions": "The initial distance of the ego vehicle to the intersection and the adversarial vehicle to the intersection are restricted. The scenario terminates when the ego vehicle has traveled a certain distance from its spawn point."
}}

Example 2 (single ego, multiple adversarials):
{{
    "Scenario": "Ego vehicle encounters a chain of debris in its lane but continues to drive in the same lane.",
    "Ego": "A car travels forward.",
    "Adversarials": [
    "A debris object remains stationary on the road.",
    "A debris object remains stationary on the road.",
    "A debris object remains stationary on the road."
    ],
    "Spatial Relation": "The ego vehicle spawns on the centerline of a random lane. A chain of three debris objects are spawned in sequence in the ego vehicle's lane.",
    "Requirement and restrictions": "The ego vehicle must initially be a certain distance away from any intersection. The scenario terminates when the ego vehicle has passed the chain of debris and traveled a certain distance from its spawn point."
}}

Extract and convert the following scenario into the exact JSON format requested. Output ONLY the JSON object.

Scenic Code:
{scenic_code}

Description Text:
{description_text}