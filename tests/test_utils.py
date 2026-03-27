def test_video_recording():
    scenic_code = """# Header Settings:
description = "Null description"
param map = localPath('../../maps/Town07.xodr')
param carla_map = 'Town07'
model scenic.simulators.carla.model
MODEL = 'vehicle.lincoln.mkz_2017'
param weather = 'MidRainyNoon'
# End of Header Settings

# Parameters for scenario setup
param OPT_EGO_SPEED = Range(1, 5)
param OPT_BRAKE_DISTANCE = Range(5, 8)
param ADV_SPEED = Range(7, 10) # Used for both AdvPerpendicular and AdvBehind
param OPT_ADV_BEHIND_DIST = Range(10, 20) # Distance AdvBehind is behind ego

# Intersection and Lane selection
# Fix: The original code attempted to iterate over 'intersection' which was a Uniform distribution.
# We need to first filter and then sample a concrete intersection object.
validIntersections = list(filter(lambda i: i.is4Way and i.isSignalized, network.intersections))
require len(validIntersections) > 0 # Ensure such an intersection configuration exists
intersection = Uniform(*validIntersections)

# To satisfy "AdvBehind" being behind ego and turning left,
# and "Spatial Relation" implying an adjacent lane interaction,
# we select an ego lane that has an adjacent left-turn lane.
potentialEgoLanePairs = []
for lane in intersection.incomingLanes:
    if any(m.type is ManeuverType.STRAIGHT for m in lane.maneuvers):
        # Check for a left-adjacent lane that also has a LEFT_TURN maneuver at the same intersection
        if lane.section._laneToLeft is not None and            lane.section._laneToLeft.lane in intersection.incomingLanes and            any(m.type is ManeuverType.LEFT_TURN for m in lane.section._laneToLeft.lane.maneuvers):
            potentialEgoLanePairs.append((lane, lane.section._laneToLeft.lane))
        # Check for a right-adjacent lane that also has a LEFT_TURN maneuver at the same intersection
        elif lane.section._laneToRight is not None and              lane.section._laneToRight.lane in intersection.incomingLanes and              any(m.type is ManeuverType.LEFT_TURN for m in lane.section._laneToRight.lane.maneuvers):
            potentialEgoLanePairs.append((lane, lane.section._laneToRight.lane))

require len(potentialEgoLanePairs) > 0 # Ensure such a lane configuration exists

egoInitLane, advBehindInitLane = Uniform(*potentialEgoLanePairs)

# Ego vehicle setup: proceeds straight
egoManeuver = Uniform(*filter(lambda m: m.type is ManeuverType.STRAIGHT and m.startLane is egoInitLane, intersection.maneuvers))
egoTrajectory = [egoInitLane, egoManeuver.connectingLane, egoManeuver.endLane]
egoSpawnPt = new OrientedPoint in egoInitLane.centerline

# Adversary 1 (AdvPerpendicular): approaches from a perpendicular road and makes a left turn
advPerpendicularManeuver = Uniform(*filter(lambda m: m.type is ManeuverType.LEFT_TURN, egoManeuver.conflictingManeuvers))
advPerpendicularInitLane = advPerpendicularManeuver.startLane
advPerpendicularTrajectory = [advPerpendicularInitLane, advPerpendicularManeuver.connectingLane, advPerpendicularManeuver.endLane]
advPerpendicularSpawnPt = new OrientedPoint in advPerpendicularInitLane.centerline

# Adversary 2 (AdvBehind): positioned behind the ego vehicle (in an adjacent lane) and makes a left turn
advBehindManeuver = Uniform(*filter(lambda m: m.type is ManeuverType.LEFT_TURN and m.startLane is advBehindInitLane, intersection.maneuvers))
advBehindTrajectory = [advBehindInitLane, advBehindManeuver.connectingLane, advBehindManeuver.endLane]

# Place AdvBehindSpawnPt behind egoSpawnPt in its adjacent lane
# Using the pattern from few-shot "Spatial Relation" examples
adjLanePt = advBehindInitLane.centerline.project(egoSpawnPt.position)
advBehindSpawnPt = new OrientedPoint following roadDirection from adjLanePt for -globalParameters.OPT_ADV_BEHIND_DIST

# Behaviors
behavior EgoBehavior():
    try:
        do FollowTrajectoryBehavior(trajectory=egoTrajectory, target_speed=globalParameters.OPT_EGO_SPEED)
    interrupt when (withinDistanceToObjsInLane(self, globalParameters.OPT_BRAKE_DISTANCE)):
        take SetThrottleAction(0)
        take SetBrakeAction(1)

behavior AdversaryBehavior(trajectory):
    do FollowTrajectoryBehavior(target_speed=globalParameters.ADV_SPEED, trajectory=trajectory)

# Agents
ego = new Car at egoSpawnPt,
    with rolename 'hero',
    with blueprint MODEL,
    with behavior EgoBehavior()

advPerpendicular = new Car at advPerpendicularSpawnPt,
    with blueprint MODEL,
    with behavior AdversaryBehavior(advPerpendicularTrajectory)

advBehind = new Car at advBehindSpawnPt,
    with blueprint MODEL,
    with behavior AdversaryBehavior(advBehindTrajectory)

# Requirements for initial distances
param EGO_INIT_DIST = Range(30, 40) # From few-shot example for ego going straight
param ADV1_INIT_DIST = Range(10, 20) # From few-shot example for conflicting left turn

require EGO_INIT_DIST[0] <= (distance from egoSpawnPt to intersection) <= EGO_INIT_DIST[1]
require ADV1_INIT_DIST[0] <= (distance from advPerpendicularSpawnPt to intersection) <= ADV1_INIT_DIST[1]
# For AdvBehind, ensure it's positioned before the intersection
require (distance from advBehindSpawnPt to intersection) > 0

# Termination
TERM_DIST = 100
terminate when (distance from ego to egoSpawnPt) > TERM_DIST

# Monitor TrafficLights
monitor TrafficLights():
    freezeTrafficLights()
    while True:
        # Requirement: ego light initially red, then green.
        # This implementation turns ego green when it's close, implicitly assuming it was red before.
        if withinDistanceToTrafficLight(ego, 100):
            setClosestTrafficLightStatus(ego, "green")
        # AdvPerpendicular is conflicting with ego (ego straight, adv left turn), so its light should be red.
        if withinDistanceToTrafficLight(advPerpendicular, 100):
            setClosestTrafficLightStatus(advPerpendicular, "red")
        # AdvBehind is behind ego, turning left. If ego is straight (green),
        # an adjacent left-turn lane might also be green or have a protected turn.
        # For simplicity and to allow its maneuver, set its light to green.
        if withinDistanceToTrafficLight(advBehind, 100):
            setClosestTrafficLightStatus(advBehind, "green")
        wait

require monitor TrafficLights()
    """
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.utils.helpers import run_simulation_in_carla_and_save_video

    video_path = run_simulation_in_carla_and_save_video(scenic_code, "test_scenario")
    if not video_path:
        print("Failed to run simulation")
        exit(1)
    print(f"Video saved to {video_path}")

    ### Run the script manually: src/utils/run_scenic_batch.sh temp_scenic_code --outdir temp/video --logdir temp/logs

def test_get_error_message_from_logs():
    from src.utils.helpers import get_error_message_from_logs
    error_message = get_error_message_from_logs("test_scenario")
    print(error_message)

def test_get_scenario_document_with_scenario_id():
    from src.utils.helpers import get_scenario_document_with_scenario_id
    scenario_document = get_scenario_document_with_scenario_id("CARLA_Leaderboard_2")
    print(scenario_document)

if __name__ == "__main__":
    test_video_recording()
    # test_get_error_message_from_logs()
    # test_get_scenario_document_with_scenario_id()