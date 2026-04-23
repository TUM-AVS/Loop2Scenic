from pathlib import Path
import csv

SOURCE_PATH = Path("/home/dellpro2/chenli/ads-mrag/ads-mrag/data/scenarios")
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

if __name__ == "__main__":
    csv_path = collect_description()
    print(f"Wrote scenario descriptions to: {csv_path}")