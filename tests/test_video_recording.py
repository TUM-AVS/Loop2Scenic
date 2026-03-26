if __name__ == "__main__":
    scenic_code = """description = "Ego vehicle loses control on bad road conditions and recovers to its original lane."
param map = localPath('../../../1data/maps/Town05.xodr')
param carla_map = 'Town05'
model scenic.simulators.carla.model
MODEL = 'vehicle.lincoln.mkz_2017'
param weather = 'WetNoon'

egoInitLane = Uniform(*network.lanes)
egoSpawnPt = new OrientedPoint in egoInitLane.centerline

param OPT_EGO_SPEED = Range(7, 10)
param LOSS_CONTROL_STEER = 0.4
param LOSS_CONTROL_DURATION = 35

behavior EgoBehavior(speed, target_lane_sec):
    # Phase 1: Normal driving
    do FollowLaneBehavior(target_speed=speed) for Range(2, 4) seconds
    
    # Phase 2: Loss of control (simulated by a sustained steering offset)
    count = 0
    while count < globalParameters.LOSS_CONTROL_DURATION:
        take SetSteerAction(globalParameters.LOSS_CONTROL_STEER), SetThrottleAction(0.5)
        count = count + 1
    
    # Phase 3: Recovery (return to the original lane section)
    do LaneChangeBehavior(laneSectionToSwitch=target_lane_sec, target_speed=speed)
    do FollowLaneBehavior(target_speed=speed)

egoLaneSec = egoInitLane.sections[0]

ego = new Car at egoSpawnPt,
    with rolename 'hero',
    with blueprint MODEL,
    with behavior EgoBehavior(globalParameters.OPT_EGO_SPEED, egoLaneSec)

require egoSpawnPt not in network.intersections
require (distance from egoSpawnPt to intersection) > 30
terminate after 25 seconds
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