def _project_root():
    import os
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_video_recording():
    import os
    import sys

    scenic_code = ""
    with open("tests/scenic_code.txt", "r") as f:
        scenic_code = f.read()

    sys.path.append(_project_root())
    from src.utils.helpers import run_simulation_in_carla_and_save_video

    video_path = run_simulation_in_carla_and_save_video(scenic_code, "test_scenario")
    if not video_path:
        print("Failed to run simulation")
        exit(1)
    print(f"Video saved to {video_path}")

    ### Run the script manually: src/utils/run_scenic_batch.sh temp_scenic_code --outdir temp/video --logdir temp/logs


def test_run_scenarios_rerun():
    import sys
    from pathlib import Path

    sys.path.append(_project_root())
    from src.utils.helpers import run_simulation_in_carla_and_save_video

    scenarios_dir = Path(_project_root()) / "data" / "scenarios_rerun_failed"
    scenic_files = sorted(scenarios_dir.glob("*.scenic"))

    if not scenic_files:
        print(f"No scenic files found in {scenarios_dir}")
        return

    failed = []
    for scenic_path in scenic_files:
        scenario_id = scenic_path.stem
        scenic_code = scenic_path.read_text()
        print(f"Running scenario: {scenario_id}")
        video_path = run_simulation_in_carla_and_save_video(scenic_code, scenario_id)
        if not video_path:
            print(f"Failed to run simulation for {scenario_id}")
            failed.append(scenario_id)
        else:
            print(f"Video saved to {video_path}")

    if failed:
        print(f"Failed scenarios ({len(failed)}): {failed}")
        exit(1)
    print(f"All {len(scenic_files)} scenarios completed successfully")

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
    test_run_scenarios_rerun()
    # test_video_recording()
    # test_get_error_message_from_logs()
    # test_get_scenario_document_with_scenario_id()
    # test_scenic_grammar()