"""
Script to ingest scenarios into the vector store.

Usage:
    python scripts/ingest_documents.py --source <directory_path>
"""

import csv
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import RAGPipeline
from src.utils import setup_logging

SOURCE_PATH = Path("/home/dellpro2/chenli/ads-mrag/ads-mrag/data/scenarios")


def scan_test_subfolders_for_required_files(
    base_dir: Path | str = Path("/home/dellpro2/chenli/ads-mrag/ads-mrag/data/scenarios"),
) -> dict[str, list[str]]:
    """
    Check each immediate subfolder in base_dir for required files.

    Returns:
        Dict mapping subfolder name -> list of missing required filenames.
        Subfolders with an empty list are complete.
    """
    base_path = Path(base_dir)
    required_files = ("new_description.txt", "video.mp4")

    if not base_path.exists():
        raise FileNotFoundError(f"Base directory does not exist: {base_path}")
    if not base_path.is_dir():
        raise NotADirectoryError(f"Base path is not a directory: {base_path}")

    results: dict[str, list[str]] = {}

    for subfolder in sorted(base_path.iterdir()):
        if not subfolder.is_dir():
            continue

        missing_files = [
            filename for filename in required_files if not (subfolder / filename).is_file()
        ]
        results[subfolder.name] = missing_files

    print(results)


def step_1_interpret_and_extract(pipeline: RAGPipeline) -> list[dict]:
    """Step 1: interpret (optional) and extract scenarios from SOURCE_PATH."""
    print(f"Ingesting scenarios from: {SOURCE_PATH}")
    print("Step 1: Interpreting scenarios with VLM...")
    pipeline.interpret_scenarios(directory_path=SOURCE_PATH)
    scenarios_dicts = pipeline.multimodal_interpreter.extract_from_directory(
        SOURCE_PATH, use_new_description=True
    )
    return scenarios_dicts


def step_2_embed_and_add_one_by_one(
    pipeline: RAGPipeline, scenarios_dicts: list[dict]
) -> tuple[list[str], list[dict[str, str]]]:
    """Step 2: embed one scenario, then add it to vector DB."""
    print("Step 2: Embedding and adding scenarios one-by-one...")
    failure_rows: list[dict[str, str]] = []
    all_doc_ids: list[str] = []

    for idx, scenario in enumerate(scenarios_dicts):
        folder_path = scenario.get("folder_path", "")
        folder_name = Path(folder_path).name if folder_path else ""
        scenario_name = str(
            scenario.get("scenario_name")
            or scenario.get("name")
            or scenario.get("id")
            or ""
        )

        try:
            embedded_doc = pipeline.embed_one_scenario(scenario)
        except Exception as exc:
            failure_rows.append(
                {
                    "index": str(idx),
                    "folder_name": folder_name,
                    "scenario_name": scenario_name,
                    "stage": "embedding",
                    "error": str(exc),
                }
            )
            print(
                f"[{idx}] Embedding failed for folder='{folder_name}' "
                f"scenario='{scenario_name}': {exc}"
            )
            continue

        if not embedded_doc:
            failure_rows.append(
                {
                    "index": str(idx),
                    "folder_name": folder_name,
                    "scenario_name": scenario_name,
                    "stage": "embedding",
                    "error": "embed_one_scenario returned no document",
                }
            )
            print(
                f"[{idx}] Embedding returned no document for "
                f"folder='{folder_name}' scenario='{scenario_name}'"
            )
            continue

        try:
            doc_ids = pipeline.add_documents_to_vector_store([embedded_doc])
            all_doc_ids.extend(doc_ids)
        except Exception as exc:
            failure_rows.append(
                {
                    "index": str(idx),
                    "folder_name": folder_name,
                    "scenario_name": scenario_name,
                    "stage": "vector_store",
                    "error": str(exc),
                }
            )
            print(
                f"[{idx}] Vector store add failed for folder='{folder_name}' "
                f"scenario='{scenario_name}': {exc}"
            )
            continue

    return all_doc_ids, failure_rows


def write_embedding_failures_csv(failure_rows: list[dict[str, str]]) -> Path:
    """Write embedding failures to logs/embedding_failures.csv."""
    logs_dir = Path(__file__).parent.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    failures_csv_path = logs_dir / "embedding_failures.csv"

    with failures_csv_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile, fieldnames=["index", "folder_name", "scenario_name", "stage", "error"]
        )
        writer.writeheader()
        writer.writerows(failure_rows)

    return failures_csv_path


def get_immediate_subfolder_names(folder_path: Path | str) -> list[str]:
    """
    Get all immediate subfolder names under a folder.

    Args:
        folder_path: Parent folder to scan.

    Returns:
        Sorted list of subfolder names.
    """
    base_path = Path(folder_path)
    if not base_path.exists():
        raise FileNotFoundError(f"Folder does not exist: {base_path}")
    if not base_path.is_dir():
        raise NotADirectoryError(f"Expected a directory, got: {base_path}")

    return sorted([child.name for child in base_path.iterdir() if child.is_dir()])


def compare_subfolders_and_item_ids(
    folder_path: Path | str,
    item_ids: list[str],
) -> dict[str, list[str]]:
    """
    Compare local subfolder names and vector-store item IDs.

    Args:
        folder_path: Parent folder whose immediate subfolder names are expected IDs.
        item_ids: IDs fetched from vector DB.

    Returns:
        Dict with:
        - subfolder_names
        - item_ids
        - only_in_subfolders (missing in DB)
        - only_in_item_ids (extra in DB)
        - in_both
    """
    subfolder_names = get_immediate_subfolder_names(folder_path)

    subfolder_set = set(subfolder_names)
    item_id_set = {str(item_id) for item_id in item_ids}

    comparison = {
        "subfolder_names": sorted(subfolder_set),
        "item_ids": sorted(item_id_set),
        "only_in_subfolders": sorted(subfolder_set - item_id_set),
        "only_in_item_ids": sorted(item_id_set - subfolder_set),
        "in_both": sorted(subfolder_set & item_id_set),
    }
    return comparison


def ingest_raw_scenarios():
    # Setup logging
    setup_logging(level="INFO")
    
    # Validate source is a directory
    if not SOURCE_PATH.exists():
        print(f"Error: Path does not exist: {SOURCE_PATH}")
        sys.exit(1)
    
    if not SOURCE_PATH.is_dir():
        print(f"Error: Source must be a directory: {SOURCE_PATH}")
        sys.exit(1)
    
    # Initialize pipeline
    print("Initializing RAG Pipeline...")
    pipeline = RAGPipeline()
    
    # scenarios_dicts = step_1_interpret_and_extract(pipeline)
    # if not scenarios_dicts:
    #     print("No scenarios found to process.")
    #     return

    scenarios_dicts = pipeline.multimodal_interpreter.extract_from_directory(
        SOURCE_PATH, use_new_description=True
    )

    print(f"The first scenario dictionary: {scenarios_dicts[0]}")

    all_doc_ids, failure_rows = step_2_embed_and_add_one_by_one(pipeline, scenarios_dicts)
    failures_csv_path = write_embedding_failures_csv(failure_rows)

    print(f"\n✓ Successfully added {len(all_doc_ids)} documents to vector store")
    print(f"Embedding failures: {len(failure_rows)}")
    print(f"Embedding failure report saved to: {failures_csv_path}")

def vector_db_operations():
    pipeline = RAGPipeline()
    stats = pipeline.get_stats()
    print(f"The stats: {stats}")

    item_ids = pipeline.vectorstore.get_all_item_ids(collection_name="scenarios", page_size=200)
    print(f"The item ids: {item_ids}")

    comparison = compare_subfolders_and_item_ids(SOURCE_PATH, item_ids)
    print(f"Total subfolders: {len(comparison['subfolder_names'])}")
    print(f"Total item IDs: {len(comparison['item_ids'])}")
    print(f"Only in subfolders (missing in DB): {comparison['only_in_subfolders']}")
    print(f"Only in item IDs (extra in DB): {comparison['only_in_item_ids']}")


if __name__ == "__main__":
    vector_db_operations()
    # ingest_raw_scenarios()
    # scan_test_subfolders_for_required_files()