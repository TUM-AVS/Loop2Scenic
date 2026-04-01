def test_video_recording():
    scenic_code = """from scenic.domains.driving.roads import ManeuverType

param map = localPath('../../maps/BEL_Brussels-50_9_T-1.xodr')
param carla_map = None
model scenic.simulators.carla.model

MODEL = 'vehicle.lincoln.mkz_2017'
param weather = 'ClearNoon'

# 1. Find intersection and left turn maneuver
intersection = Uniform(*filter(lambda i: any(m.type is ManeuverType.LEFT_TURN for m in i.maneuvers), network.intersections))

egoManeuver = Uniform(*filter(lambda m: m.type is ManeuverType.LEFT_TURN, intersection.maneuvers))
egoInitLane = egoManeuver.startLane
egoTrajectory = [egoInitLane, egoManeuver.connectingLane, egoManeuver.endLane]
egoSpawnPt = new OrientedPoint in egoInitLane.centerline

param OPT_EGO_SPEED = Range(3, 5)

# 2. Ego Behavior: Follow the full path
behavior EgoBehavior():
    do FollowTrajectoryBehavior(trajectory=egoTrajectory, target_speed=globalParameters.OPT_EGO_SPEED)

# 3. Spawn the Ego Car
ego = new Car at egoSpawnPt,
    with rolename 'hero',
    with blueprint MODEL,
    with behavior EgoBehavior()

# (Keeping the strict require statement commented out to prevent the 2000 RejectionException we fixed earlier)
# EGO_INIT_DIST = [10, 15]
# require EGO_INIT_DIST[0] <= (distance from egoSpawnPt to intersection) <= EGO_INIT_DIST[1]

# 4. NEW TERMINATION CONDITION
terminate when (ego in egoManeuver.endLane) and (distance from ego to intersection > 3)
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