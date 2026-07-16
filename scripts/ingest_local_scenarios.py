"""
Ingest local scenario folders into Milvus using the configured Qwen embedder.

Skips the VLM interpretation step entirely — new_description.txt / new_description.json
already exist in each scenario folder, so ingestion is deterministic and API-free.
Documents are embedded as text+video (matching the production multimodal index).

Usage:
    python scripts/ingest_local_scenarios.py [--folder data/scenarios] [--reset]
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.ingestion import MultimodalDocumentInterpreter
from src.services import MilvusVectorStore, get_embedder
from src.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest scenarios into local Milvus (no VLM)")
    parser.add_argument("--folder", default="data/scenarios", help="Scenario folder root")
    parser.add_argument("--config", default=None, help="Config yaml path (controls embedder + collection)")
    parser.add_argument("--doc-modality", choices=["text_video", "text_only", "video_only"], default="text_video",
                        help="What each document embeds: text+video (default), text only, or video only")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate the collection first")
    args = parser.parse_args()

    config = get_config(args.config)
    setup_logging(level=config.logging.level, log_format=config.logging.format)
    logger = logging.getLogger("ingest_local")

    interpreter = MultimodalDocumentInterpreter()
    docs = interpreter.extract_from_directory(args.folder, use_new_description=True)
    logger.info("Extracted %d scenario documents from %s", len(docs), args.folder)
    if not docs:
        raise SystemExit("No documents extracted — check the folder path")

    embedder_kwargs = {}
    if getattr(config.embedding, "model_path", None):
        embedder_kwargs["model_path"] = config.embedding.model_path
    embedder = get_embedder(
        provider=config.embedding.provider,
        model_name=config.embedding.model_name,
        **embedder_kwargs,
    )
    logger.info("Embedder ready (dim=%d)", embedder.dimension)

    store = MilvusVectorStore(
        embedding_dim=embedder.dimension,
        collection_name=config.vector_db.collection_name,
        snippets_collection_name=config.vector_db.snippets_collection_name,
        connection_args={"host": config.vector_db.host, "port": config.vector_db.port},
        index_params={
            "metric_type": config.vector_db.distance_metric.upper(),
            "index_type": config.vector_db.index_type,
            "params": {"nlist": config.vector_db.nlist},
        },
        search_params={
            "metric_type": config.vector_db.distance_metric.upper(),
            "params": {"nprobe": config.vector_db.nprobe},
        },
    )
    if args.reset:
        store.reset_collection()
        store._init_collection()

    ok = 0
    failures: list[tuple[str, str]] = []
    for i, doc in enumerate(docs):
        scenario_id = Path(doc["folder_path"]).name
        try:
            payload = {
                "text": "" if args.doc_modality == "video_only" else doc.get("description", ""),
                "image": doc.get("image", ""),
                "video": "" if args.doc_modality == "text_only" else doc.get("video", ""),
            }
            doc["embedding"] = embedder.encode([payload])[0]
            doc["metadata"] = doc.get("description_json", {})
            store.add_documents([doc])
            ok += 1
            logger.info("[%d/%d] ingested %s", i + 1, len(docs), scenario_id)
        except Exception as exc:
            failures.append((scenario_id, str(exc)))
            logger.error("[%d/%d] FAILED %s: %s", i + 1, len(docs), scenario_id, exc)

    logger.info("Ingestion complete: %d ok, %d failed", ok, len(failures))
    if failures:
        logger.error("Failed scenarios: %s", [f[0] for f in failures])
    print(f"INGEST_DONE ok={ok} failed={len(failures)}")


if __name__ == "__main__":
    main()
