from pathlib import Path
import csv
import re
import subprocess
import tempfile
import shutil

SOURCE_PATH = Path("/home/dellpro2/chenli/ads-mrag/ads-mrag/data/chat2scenic")
EVAL_PATH = Path("/home/dellpro2/chenli/ads-mrag/ads-mrag/data/eval")

def collect_description(
    folder_path: Path = SOURCE_PATH,
    output_csv_path: Path = EVAL_PATH / "scenario_descriptions.csv",
) -> Path:
    """
    Collect scenario IDs and description.txt content into a CSV file.

    CSV columns:
    - scenario_id: subfolder name
    - description: text content from description.txt
    """
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Expected directory, got: {folder_path}")

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    for subfolder in sorted(folder_path.iterdir()):
        if not subfolder.is_dir():
            continue

        scenario_id = subfolder.name
        description_file = subfolder / "description.txt"

        if description_file.is_file():
            description_text = description_file.read_text(encoding="utf-8").strip()
        else:
            description_text = ""

        rows.append({"scenario_id": scenario_id, "description": description_text})

    with output_csv_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["scenario_id", "description"])
        writer.writeheader()
        writer.writerows(rows)

    return output_csv_path

def reformat_scenarios(
    folder_path: Path = SOURCE_PATH,
) -> list[Path]:
    """
    Reformat flat .scenic files into scenario folders.

    For each *.scenic file directly under folder_path:
    1) Create a folder named after the scenic file stem.
    2) Move the file into that folder.
    3) Rename it to code.scenic.

    Returns:
        List of final file paths for moved scenic files.
    """
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Expected directory, got: {folder_path}")

    moved_files: list[Path] = []
    scenic_files = sorted(folder_path.glob("*.scenic"))

    for scenic_file in scenic_files:
        scenario_name = scenic_file.stem
        scenario_folder = folder_path / scenario_name
        scenario_folder.mkdir(parents=True, exist_ok=True)

        target_file = scenario_folder / "code.scenic"
        if target_file.exists():
            raise FileExistsError(f"Target already exists, refusing to overwrite: {target_file}")

        scenic_file.rename(target_file)
        moved_files.append(target_file)

    return moved_files

def extract_description(
    folder_path: SOURCE_PATH,
) -> list[Path]:
    """
    Extract description from each subfolder's code.scenic and write description.txt.

    Expected pattern inside code.scenic: description="..."
    """
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Expected directory, got: {folder_path}")

    written_files: list[Path] = []
    pattern = re.compile(r'description\s*=\s*"([^"]*)"')

    for subfolder in sorted(folder_path.iterdir()):
        if not subfolder.is_dir():
            continue

        scenic_file = subfolder / "code.scenic"
        if not scenic_file.is_file():
            continue

        content = scenic_file.read_text(encoding="utf-8")
        match = pattern.search(content)
        if not match:
            continue

        description = match.group(1)
        description_file = subfolder / "description.txt"
        description_file.write_text(description, encoding="utf-8")
        written_files.append(description_file)

    return written_files

def run_simulation_and_save_video(
    folder_path: Path = SOURCE_PATH,
) -> list[Path]:
    """
    Run simulation for each subfolder and save videos/logs in-place.

    For each immediate subfolder under folder_path:
    - read/run subfolder/code.scenic
    - save videos to subfolder/video
    - save logs to subfolder/logs

    Returns:
        List of subfolders that completed successfully.
    """
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Expected directory, got: {folder_path}")

    repo_root = Path(__file__).resolve().parent.parent
    batch_script = repo_root / "src" / "utils" / "run_scenic_batch.sh"
    recorder_script = repo_root / "src" / "utils" / "recorder_scenic.py"
    if not batch_script.is_file():
        raise FileNotFoundError(f"Batch script not found: {batch_script}")
    if not recorder_script.is_file():
        raise FileNotFoundError(f"Recorder script not found: {recorder_script}")

    # run_scenic_batch.sh does not expose recorder view args directly.
    # Use a small wrapper so each simulation records BEV+FPV+TPV.
    with tempfile.TemporaryDirectory(prefix="recorder_wrapper_") as tmp_dir:
        wrapper_path = Path(tmp_dir) / "recorder_with_all_views.py"
        wrapper_path.write_text(
            "\n".join(
                [
                    "#!/usr/bin/env python3",
                    "import subprocess",
                    "import sys",
                    f'RECORDER = r"{recorder_script}"',
                    "cmd = [sys.executable, RECORDER, *sys.argv[1:], '--views', 'bev', 'fpv', 'tpv']",
                    "raise SystemExit(subprocess.call(cmd))",
                ]
            ),
            encoding="utf-8",
        )
        wrapper_path.chmod(0o755)

        success_subfolders: list[Path] = []

        for subfolder in sorted(folder_path.iterdir()):
            if not subfolder.is_dir():
                continue

            scenic_file = subfolder / "code.scenic"
            if not scenic_file.is_file():
                continue

            video_dir = subfolder / "video"
            log_dir = subfolder / "logs"
            video_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)

            result = subprocess.run(
                [
                    str(batch_script),
                    "--recorder",
                    str(wrapper_path),
                    "--outdir",
                    str(video_dir),
                    "--logdir",
                    str(log_dir),
                    str(scenic_file),
                ],
                capture_output=True,
                text=True,
                cwd=str(repo_root),
            )

            if result.returncode != 0:
                print(f"[FAILED] {subfolder.name}: {result.stderr.strip()}")
                continue

            expected_videos = [video_dir / "BEV.mp4", video_dir / "FPV.mp4", video_dir / "TPV.mp4"]
            if not all(path.exists() for path in expected_videos):
                print(f"[FAILED] {subfolder.name}: missing expected videos in {video_dir}")
                continue

            success_subfolders.append(subfolder)

    return success_subfolders


def move_videos(folder_path: Path = SOURCE_PATH) -> list[Path]:
    """
    Copy each subfolder's video/BEV.mp4 to the subfolder root.

    For each immediate subfolder under folder_path:
    - source: subfolder/video/BEV.mp4
    - target: subfolder/BEV.mp4

    Returns:
        List of target BEV paths that were created/updated.
    """
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Expected directory, got: {folder_path}")

    copied_files: list[Path] = []
    for subfolder in sorted(folder_path.iterdir()):
        if not subfolder.is_dir():
            continue

        source_bev = subfolder / "video" / "BEV.mp4"
        if not source_bev.is_file():
            continue

        target_bev = subfolder / "BEV.mp4"
        shutil.copy2(source_bev, target_bev)
        copied_files.append(target_bev)

    return copied_files


def remove_extra_folders(folder_path: Path = SOURCE_PATH) -> list[Path]:
    """
    Remove `video` and `logs` folders under each immediate subfolder in folder_path.

    Returns:
        List of removed folder paths.
    """
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Expected directory, got: {folder_path}")

    removed: list[Path] = []
    for subfolder in sorted(folder_path.iterdir()):
        if not subfolder.is_dir():
            continue

        for folder_name in ("video", "logs"):
            target_dir = subfolder / folder_name
            if not target_dir.is_dir():
                continue
            shutil.rmtree(target_dir)
            removed.append(target_dir)

        # remove the video.mp4 file
        for file_name in ("video.mp4", "new_description.json", "new_description.txt"):
            file_path = subfolder / file_name
            if file_path.is_file():
                file_path.unlink()
                removed.append(file_path)

    return removed

def check_if_all_files_exist(folder_path: Path = SOURCE_PATH) -> bool:
    """
    Check required files in each immediate subfolder.

    Required files per subfolder:
    - BEV.mp4
    - code.scenic
    - description.txt
    - new_description.json
    - new_description.txt

    Prints subfolder name and missing file names for incomplete subfolders.
    """
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Expected directory, got: {folder_path}")

    required_files = (
        "BEV.mp4",
        "code.scenic",
        "description.txt",
        "new_description.json",
        "new_description.txt",
    )

    all_complete = True
    for subfolder in sorted(folder_path.iterdir()):
        if not subfolder.is_dir():
            continue

        missing = [
            filename
            for filename in required_files
            if not (subfolder / filename).is_file()
        ]
        if missing:
            all_complete = False
            print(f"{subfolder.name}: missing {', '.join(missing)}")

    return all_complete


def rename_png_to_image_png(folder_path: Path) -> list[Path]:
    """
    Rename each subfolder's PNG file to image.png.

    For each immediate subfolder under folder_path:
    - If image.png already exists, skip that subfolder.
    - Otherwise rename a single .png file to image.png.
      When multiple PNGs exist, prefer ``<subfolder_name>.png``; otherwise use the
      only PNG present, or the first sorted match if ambiguous.

    Returns:
        List of paths to the created image.png files.
    """
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Expected directory, got: {folder_path}")

    renamed_files: list[Path] = []

    for subfolder in sorted(folder_path.iterdir()):
        if not subfolder.is_dir():
            continue

        target = subfolder / "image.png"
        if target.is_file():
            continue

        png_files = sorted(subfolder.glob("*.png"))
        if not png_files:
            print(f"[SKIP] {subfolder.name}: no .png file found")
            continue

        preferred = subfolder / f"{subfolder.name}.png"
        if preferred.is_file():
            source = preferred
        elif len(png_files) == 1:
            source = png_files[0]
        else:
            source = png_files[0]
            print(
                f"[WARN] {subfolder.name}: multiple PNGs found; "
                f"renaming {source.name} -> image.png"
            )

        source_name = source.name
        source.rename(target)
        renamed_files.append(target)
        print(f"[OK] {subfolder.name}: {source_name} -> image.png")

    return renamed_files


if __name__ == "__main__":
    # csv_path = collect_description()
    # reformat_scenarios()
    # extract_description()
    # run_simulation_and_save_video()
    # move_videos()
    # remove_extra_folders()
    rename_png_to_image_png(Path("/home/avsaw1/chenli/ads-mrag/data/inference_data/image-only"))
    # check_if_all_files_exist()