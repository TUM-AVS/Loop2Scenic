def test_video_recording():
    import os
    import sys

    scenic_code = ""
    with open("tests/scenic_code.txt", "r") as f:
        scenic_code = f.read()

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

def test_scenic_grammar():
    import os
    import sys
    from scenic import scenarioFromFile
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    scenario = scenarioFromFile(
        "tests/scenic_code.scenic",
        model="scenic.simulators.carla.model",
        mode2D=True
    )
    print("Parse/compile OK")
    scene, _ = scenario.generate(maxIterations=1)
    print("Scene generation OK")

if __name__ == "__main__":
    test_video_recording()
    # test_get_error_message_from_logs()
    # test_get_scenario_document_with_scenario_id()
    # test_scenic_grammar()