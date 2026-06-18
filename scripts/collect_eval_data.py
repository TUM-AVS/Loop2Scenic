from pathlib import Path
import csv
import random
import re
import subprocess
import sys
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.utils.simulation import build_run_scenic_batch_command, repo_root, resolve_path

SOURCE_PATH = Path("/home/dellpro2/chenli/ads-mrag/ads-mrag/data/chat2scenic")
EVAL_PATH = Path("/home/dellpro2/chenli/ads-mrag/ads-mrag/data/eval")
REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_272_PATH = REPO_ROOT / "data" / "272eval"
INFERENCE_TEXT_ONLY_PATH = REPO_ROOT / "data" / "inference_data" / "text-only"
INFERENCE_TEXT_IMAGE_PATH = REPO_ROOT / "data" / "inference_data" / "text-image"
INFERENCE_IMAGE_ONLY_PATH = REPO_ROOT / "data" / "inference_data" / "image-only"
SCENARIOS_PATH = REPO_ROOT / "data" / "scenarios"
WENTING100_PATH = REPO_ROOT / "data" / "wenting100"

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

    project_root = repo_root()
    config = get_config()
    batch_script = Path(
        resolve_path(config.simulation.batch_script)
        or project_root / "src/utils/run_scenic_batch.sh"
    )
    recorder_script = Path(
        resolve_path(config.simulation.recorder_script)
        or project_root / "src/utils/recorder_scenic.py"
    )
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

            cmd = build_run_scenic_batch_command(
                config,
                scenic_file,
                outdir=video_dir,
                logdir=log_dir,
                recorder_py=wrapper_path,
            )
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(project_root),
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


def sample_inference_data(
    source_path: Path = EVAL_272_PATH,
    text_only_path: Path = INFERENCE_TEXT_ONLY_PATH,
    text_image_path: Path = INFERENCE_TEXT_IMAGE_PATH,
    text_only_count: int = 27,
    text_image_count: int = 50,
    seed: int | None = None,
) -> dict[str, list[str]]:
    """
    Randomly sample scenario subfolders and copy them into inference_data splits.

    Selects ``text_only_count + text_image_count`` subfolders from source_path,
    then copies the first group to text_only_path and the second to text_image_path.
    """
    total_count = text_only_count + text_image_count

    if not source_path.exists():
        raise FileNotFoundError(f"Folder does not exist: {source_path}")
    if not source_path.is_dir():
        raise NotADirectoryError(f"Expected directory, got: {source_path}")

    subfolders = sorted(path for path in source_path.iterdir() if path.is_dir())
    if len(subfolders) < total_count:
        raise ValueError(
            f"Need at least {total_count} subfolders in {source_path}, found {len(subfolders)}"
        )

    rng = random.Random(seed)
    selected = rng.sample(subfolders, total_count)
    text_only_folders = selected[:text_only_count]
    text_image_folders = selected[text_only_count:]

    text_only_path.mkdir(parents=True, exist_ok=True)
    text_image_path.mkdir(parents=True, exist_ok=True)

    copied_text_only: list[str] = []
    for folder in text_only_folders:
        target = text_only_path / folder.name
        shutil.copytree(folder, target, dirs_exist_ok=True)
        copied_text_only.append(folder.name)

    copied_text_image: list[str] = []
    for folder in text_image_folders:
        target = text_image_path / folder.name
        shutil.copytree(folder, target, dirs_exist_ok=True)
        copied_text_image.append(folder.name)

    return {
        "text_only": copied_text_only,
        "text_image": copied_text_image,
    }


def sample_eval_subfolders_excluding_existing(
    source_path: Path = EVAL_272_PATH,
    text_only_path: Path = INFERENCE_TEXT_ONLY_PATH,
    text_image_path: Path = INFERENCE_IMAGE_ONLY_PATH,
    dest_path: Path = INFERENCE_TEXT_IMAGE_PATH,
    sample_count: int = 1,
    seed: int | None = None,
) -> list[str]:
    """
    Randomly sample subfolders from source_path, excluding names already present
    in text_only_path or text_image_path, then copy them to dest_path.
    """
    if not source_path.exists():
        raise FileNotFoundError(f"Folder does not exist: {source_path}")
    if not source_path.is_dir():
        raise NotADirectoryError(f"Expected directory, got: {source_path}")

    existing_names: set[str] = set()
    for folder_path in (text_only_path, text_image_path):
        if not folder_path.exists():
            continue
        for subfolder in folder_path.iterdir():
            if subfolder.is_dir():
                existing_names.add(subfolder.name)

    candidates = sorted(
        path
        for path in source_path.iterdir()
        if path.is_dir() and path.name not in existing_names
    )
    if len(candidates) < sample_count:
        raise ValueError(
            f"Need at least {sample_count} available subfolders in {source_path}, "
            f"found {len(candidates)} after excluding {len(existing_names)} existing names"
        )

    rng = random.Random(seed)
    selected = rng.sample(candidates, sample_count)

    dest_path.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for folder in selected:
        target = dest_path / folder.name
        shutil.copytree(folder, target, dirs_exist_ok=True)
        copied.append(folder.name)

    return copied


def create_text_only_folders_from_descriptions(
    descriptions_file: Path = INFERENCE_TEXT_ONLY_PATH / "descriptions.txt",
    output_path: Path = INFERENCE_TEXT_ONLY_PATH,
) -> list[Path]:
    """
    Create text-only scenario subfolders from a descriptions.txt file.

    Each non-empty line is expected in ``scenario_id;description`` format.
    Creates ``output_path / scenario_id / description.txt`` with the full line text.
    """
    if not descriptions_file.is_file():
        raise FileNotFoundError(f"Descriptions file does not exist: {descriptions_file}")

    output_path.mkdir(parents=True, exist_ok=True)
    created_folders: list[Path] = []

    for line in descriptions_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue

        scenario_id = line.split(";", 1)[0].strip()
        if not scenario_id:
            continue

        scenario_folder = output_path / scenario_id
        scenario_folder.mkdir(parents=True, exist_ok=True)
        (scenario_folder / "description.txt").write_text(line, encoding="utf-8")
        created_folders.append(scenario_folder)

    return created_folders


def count_subfolders(folder_path: Path) -> int:
    """
    Count immediate subfolders under folder_path.

    Returns:
        Number of direct child directories.
    """
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Expected directory, got: {folder_path}")

    return sum(1 for path in folder_path.iterdir() if path.is_dir())


def keep_only_txt_files(folder_path: Path = INFERENCE_TEXT_ONLY_PATH) -> list[Path]:
    """
    Delete all non-.txt files inside each immediate subfolder of folder_path.

    Returns:
        List of deleted file paths.
    """
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Expected directory, got: {folder_path}")

    deleted_files: list[Path] = []
    for subfolder in sorted(folder_path.iterdir()):
        if not subfolder.is_dir():
            continue

        for file_path in subfolder.iterdir():
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() == ".txt":
                continue

            file_path.unlink()
            deleted_files.append(file_path)

    return deleted_files


def move_random_subfolders_to_image_only(
    source_path: Path = INFERENCE_TEXT_IMAGE_PATH,
    dest_path: Path = INFERENCE_IMAGE_ONLY_PATH,
    move_count: int = 50,
    seed: int | None = None,
) -> list[str]:
    """
    Randomly move subfolders from text-image to image-only.

    Returns:
        List of moved subfolder names.
    """
    if not source_path.exists():
        raise FileNotFoundError(f"Folder does not exist: {source_path}")
    if not source_path.is_dir():
        raise NotADirectoryError(f"Expected directory, got: {source_path}")

    subfolders = sorted(path for path in source_path.iterdir() if path.is_dir())
    if len(subfolders) < move_count:
        raise ValueError(
            f"Need at least {move_count} subfolders in {source_path}, found {len(subfolders)}"
        )

    rng = random.Random(seed)
    selected = rng.sample(subfolders, move_count)

    dest_path.mkdir(parents=True, exist_ok=True)
    moved: list[str] = []
    for folder in selected:
        target = dest_path / folder.name
        if target.exists():
            raise FileExistsError(f"Target already exists, refusing to overwrite: {target}")
        shutil.move(str(folder), str(target))
        moved.append(folder.name)

    return moved


def copy_carla_scenarios_to_wenting100(
    source_path: Path = SCENARIOS_PATH,
    dest_path: Path = WENTING100_PATH,
    prefix: str = "UN",
) -> list[str]:
    """
    Copy all subfolders starting with prefix from source_path to dest_path.

    Returns:
        List of copied subfolder names.
    """
    if not source_path.exists():
        raise FileNotFoundError(f"Folder does not exist: {source_path}")
    if not source_path.is_dir():
        raise NotADirectoryError(f"Expected directory, got: {source_path}")

    dest_path.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    for subfolder in sorted(source_path.iterdir()):
        if not subfolder.is_dir():
            continue
        if not subfolder.name.startswith(prefix):
            continue

        target = dest_path / subfolder.name
        shutil.copytree(subfolder, target, dirs_exist_ok=True)
        copied.append(subfolder.name)

    return copied


def rename_png_files_to_image(
    folder_path: Path = INFERENCE_TEXT_IMAGE_PATH,
    target_name: str = "image.png",
) -> list[Path]:
    """
    Rename .png files inside each immediate subfolder to target_name.

    Skips files already named target_name. Raises if target_name already exists
    from a different source file in the same subfolder.

    Returns:
        List of renamed file paths (new locations).
    """
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Expected directory, got: {folder_path}")

    renamed_files: list[Path] = []
    for subfolder in sorted(folder_path.iterdir()):
        if not subfolder.is_dir():
            continue

        for file_path in sorted(subfolder.iterdir()):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() != ".png":
                continue
            if file_path.name == target_name:
                continue

            target = subfolder / target_name
            if target.exists():
                raise FileExistsError(
                    f"Cannot rename {file_path} to {target}: target already exists"
                )

            file_path.rename(target)
            renamed_files.append(target)

    return renamed_files



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
    # sample_inference_data()
    # sample_eval_subfolders_excluding_existing()
    # move_random_subfolders_to_image_only()
    # copy_carla_scenarios_to_wenting100()
    rename_png_files_to_image()
    # create_text_only_folders_from_descriptions()
    # print(count_subfolders(Path("data\wenting100")))
    # keep_only_txt_files()
    # rename_png_to_image_png(Path("/home/avsaw1/chenli/ads-mrag/data/inference_data/image-only"))
    # check_if_all_files_exist()
    # sample_eval_subfolders_excluding_existing()
    pass