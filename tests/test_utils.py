def test_video_recording():
    scenic_code = """description = "Ego vehicle makes a left turn at an intersection while one adversarial vehicle goes straight and another adversarial vehicle turns left behind the ego."\nparam map = localPath(\'../../maps/Town05.xodr\')\nparam carla_map = \'Town05\'\nmodel scenic.simulators.carla.model\nMODEL = \'vehicle.lincoln.mkz_2017\'\nparam weather = \'ClearNoon\'\n\nintersection = Uniform(*filter(lambda i: any(m.type is ManeuverType.LEFT_TURN for m in i.maneuvers), network.intersections))\n\negoManeuver = Uniform(*filter(lambda m: m.type is ManeuverType.LEFT_TURN and any(cm.type is ManeuverType.STRAIGHT for cm in m.conflictingManeuvers), intersection.maneuvers))\negoInitLane = egoManeuver.startLane\negoTrajectory = [egoInitLane, egoManeuver.connectingLane, egoManeuver.endLane]\negoSpawnPt = new OrientedPoint in egoInitLane.centerline\n\n# Adversary 1: Approaches from opposite direction, goes straight\nadv1Maneuver = Uniform(*filter(lambda m: m.type is ManeuverType.STRAIGHT, egoManeuver.conflictingManeuvers))\nadv1InitLane = adv1Maneuver.startLane\nadv1Trajectory = [adv1InitLane, adv1Maneuver.connectingLane, adv1Maneuver.endLane]\nadv1SpawnPt = new OrientedPoint in adv1InitLane.centerline\n\n# Adversary 2: Positioned behind ego, makes a left turn\nparam OPT_BEHIND_DIST = Range(10, 15)\nadv2SpawnPt = new OrientedPoint following roadDirection from egoSpawnPt for -globalParameters.OPT_BEHIND_DIST\nadv2Trajectory = egoTrajectory # Adversary 2 also makes a left turn, following ego\'s path\n\nparam OPT_EGO_SPEED = Range(3, 5)\nparam OPT_EGO_YIELD_DIST = Range(15, 20)\nOPT_EGO_DECISION_DEGREE = 35 deg\n\nbehavior EgoBehavior():\n    initialDir = egoSpawnPt.heading\n    try:\n        do FollowTrajectoryBehavior(trajectory=egoTrajectory, target_speed=globalParameters.OPT_EGO_SPEED)\n    interrupt when withinDistanceToAnyCars(self, globalParameters.OPT_EGO_YIELD_DIST):\n        currentDir = self.heading\n        if (abs(currentDir - initialDir) < OPT_EGO_DECISION_DEGREE):\n            take SetThrottleAction(0)\n            take SetBrakeAction(1)\n        else:\n            do FollowTrajectoryBehavior(trajectory=egoTrajectory, target_speed=globalParameters.OPT_EGO_SPEED + 2)\n            abort\n    terminate\n\nego = new Car at egoSpawnPt,\n    with rolename \'hero\',\n    with blueprint MODEL,\n    with behavior EgoBehavior()\n\nparam ADV1_SPEED = Range(7, 10) # Speed for adversary 1\n\nbehavior Adversary1Behavior(trajectory):\n    do FollowTrajectoryBehavior(target_speed=globalParameters.ADV1_SPEED, trajectory=trajectory)\n\nadversary1 = new Car at adv1SpawnPt,\n    with blueprint MODEL,\n    with behavior Adversary1Behavior(adv1Trajectory)\n\nparam ADV2_SPEED = Range(5, 7) # Speed for adversary 2\n\nbehavior Adversary2Behavior(trajectory):\n    do FollowTrajectoryBehavior(target_speed=globalParameters.ADV2_SPEED, trajectory=trajectory)\n\nadversary2 = new Car at adv2SpawnPt,\n    with blueprint MODEL,\n    with behavior Adversary2Behavior(adv2Trajectory)\n\nEGO_INIT_DIST = [10, 15]\nADV_INIT_DIST = [15, 25] # This now refers to Adversary 1\'s initial distance\n\nTERM_DIST = 50\n\nmonitor TrafficLights():\n    freezeTrafficLights()\n    while True:\n        if withinDistanceToTrafficLight(ego, 100):\n            setClosestTrafficLightStatus(ego, "green")\n        if withinDistanceToTrafficLight(adversary1, 100):\n            setClosestTrafficLightStatus(adversary1, "green")\n        if withinDistanceToTrafficLight(adversary2, 100):\n            setClosestTrafficLightStatus(adversary2, "green")\n        wait\n\nrequire monitor TrafficLights()\nrequire EGO_INIT_DIST[0] <= (distance from egoSpawnPt to intersection) <= EGO_INIT_DIST[1]\nrequire ADV_INIT_DIST[0] <= (distance from adv1SpawnPt to intersection) <= ADV_INIT_DIST[1]\nterminate when (distance from ego to egoSpawnPt) > TERM_DIST
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