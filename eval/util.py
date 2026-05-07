from __future__ import annotations

import csv
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def move_subfolders_to_c11_temp(
    source_dirs: Iterable[Path] | None = None,
    target_dir: Path | None = None,
) -> List[Tuple[Path, Path]]:
    """
    Move all direct subfolders from source directories to data/C11_temp.

    Returns a list of (source_subfolder, moved_destination) pairs.
    """
    default_sources = [
        Path("data") / "e2e_20260428_201146_1-50",
        Path("data") / "e2e_20260429_104150_51-105",
        Path("data") / "e2e_20260429_190439_carlaleaderboard",
        Path("data") / "e2e_20260429_220952_chatscene",
    ]

    sources = list(source_dirs) if source_dirs is not None else default_sources
    destination_root = target_dir if target_dir is not None else Path("data") / "C11_temp"
    destination_root.mkdir(parents=True, exist_ok=True)

    moved: List[Tuple[Path, Path]] = []
    for src in sources:
        if not src.exists():
            print(f"Skip missing source: {src}")
            continue
        if not src.is_dir():
            print(f"Skip non-directory source: {src}")
            continue

        for child in sorted(src.iterdir()):
            if not child.is_dir():
                continue

            destination = destination_root / child.name
            if destination.exists():
                print(f"Skip existing destination folder: {destination}")
                continue

            shutil.move(str(child), str(destination))
            moved.append((child, destination))
            print(f"Moved: {child} -> {destination}")

    print(f"Done. Total moved subfolders: {len(moved)}")
    return moved


def collect_generated_videos_to_c11_temp(
    c11_temp_dir: Path | None = None,
    temp_root_dir: Path | None = None,
) -> List[str]:
    """
    For each subfolder in data/C11_temp:
    1) Find one *.scenic file and use its stem as scenario folder name.
    2) Locate data/temp/<stem>/video/BEV.mp4.
    3) If found, move it to data/C11_temp/<subfolder>/generated_video.mp4.
    4) If not found, record the subfolder name.

    Returns a list of subfolder names where BEV.mp4 was not found.
    """
    c11_root = c11_temp_dir if c11_temp_dir is not None else Path("data") / "C11_temp"
    temp_root = temp_root_dir if temp_root_dir is not None else Path("data") / "temp"

    if not c11_root.exists() or not c11_root.is_dir():
        raise FileNotFoundError(f"C11 temp directory not found: {c11_root}")
    if not temp_root.exists() or not temp_root.is_dir():
        raise FileNotFoundError(f"Temp root directory not found: {temp_root}")

    missing_subfolders: List[str] = []

    for subfolder in sorted(c11_root.iterdir()):
        if not subfolder.is_dir():
            continue

        scenic_files = sorted(subfolder.glob("*.scenic"))
        if not scenic_files:
            missing_subfolders.append(subfolder.name)
            print(f"Missing .scenic file in {subfolder.name}")
            continue

        scenario_name = scenic_files[0].stem
        source_bev = temp_root / scenario_name / "video" / "BEV.mp4"
        destination_video = subfolder / "generated_video.mp4"

        if not source_bev.exists():
            missing_subfolders.append(subfolder.name)
            print(f"Missing source video for {subfolder.name}: {source_bev}")
            continue

        if destination_video.exists():
            destination_video.unlink()

        shutil.move(str(source_bev), str(destination_video))
        print(f"Moved video: {source_bev} -> {destination_video}")

    print("Subfolders missing source video:")
    print(missing_subfolders)
    return missing_subfolders


def find_duplicate_ground_truth_records(csv_file_path: Path | str) -> Dict[str, int]:
    """
    Find duplicate non-empty ground_truth values in a CSV file.

    Returns:
        dict mapping duplicate ground_truth -> count (count > 1 only)
    """
    csv_path = Path(csv_file_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        values = [
            (row.get("ground_truth") or "").strip()
            for row in reader
            if (row.get("ground_truth") or "").strip() != ""
        ]

    counts = Counter(values)
    duplicates = {k: v for k, v in counts.items() if v > 1}

    if duplicates:
        print("Duplicate ground_truth values found:")
        for key in sorted(duplicates):
            print(f"  - {key}: {duplicates[key]}")
    else:
        print("No duplicate ground_truth values found.")

    return duplicates


def append_projected_rows(
    source_csv_path: Path | str,
    target_csv_path: Path | str,
    target_columns: List[str] | None = None,
) -> int:
    """
    Append rows from source CSV to target CSV, keeping only target columns.

    Returns the number of rows appended.
    """
    src = Path(source_csv_path)
    dst = Path(target_csv_path)
    if not src.exists():
        raise FileNotFoundError(f"Source CSV not found: {src}")
    if not dst.exists():
        raise FileNotFoundError(f"Target CSV not found: {dst}")

    with src.open("r", newline="", encoding="utf-8-sig") as src_file:
        src_reader = csv.DictReader(src_file)
        src_rows = list(src_reader)

    with dst.open("r", newline="", encoding="utf-8-sig") as dst_file:
        dst_reader = csv.DictReader(dst_file)
        dst_fieldnames = list(dst_reader.fieldnames or [])

    projected_columns = target_columns if target_columns is not None else dst_fieldnames
    if not projected_columns:
        raise ValueError("Target CSV has no header columns.")

    rows_to_append: List[Dict[str, str]] = []
    for row in src_rows:
        projected = {col: (row.get(col) or "") for col in projected_columns}
        rows_to_append.append(projected)

    with dst.open("a", newline="", encoding="utf-8") as dst_file:
        writer = csv.DictWriter(dst_file, fieldnames=projected_columns)
        writer.writerows(rows_to_append)

    print(f"Appended {len(rows_to_append)} rows from {src} to {dst}")
    return len(rows_to_append)


if __name__ == "__main__":
    # move_subfolders_to_c11_temp()
    # collect_generated_videos_to_c11_temp()
    # find_duplicate_ground_truth_records("data/C11_CP+CoT+ICL+codeICL/batch_results.csv")
    append_projected_rows(
    "eval/visual_scoring_csv/C11_CP+CoT+ICL+codeICL/NHTSA_results_with_token_usage.csv",
    "eval/visual_scoring_csv/C11_CP+CoT+ICL+codeICL/input.csv",
)

