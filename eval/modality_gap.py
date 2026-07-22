"""
Modality-gap analysis (Liang et al. 2022, "Mind the Gap", arXiv:2203.02053).

For a VL embedder, embed each scenario three ways — text-only, video-only, and joint
text+video — then quantify how far apart the modality subspaces sit. This explains WHY a
text-only query retrieves poorly against a text+video document index (our 36.3 vs 50.2 finding)
and why joint text+video retrieval wins.

Metrics reported (all on L2-normalized embeddings):
- matched_cross_modal_cos : mean_i cos(text_i, video_i)         — alignment of a scenario's own
                                                                   text vs video embedding
- within_text_cos         : mean_{i!=j} cos(text_i, text_j)      — within-modality baseline
- within_video_cos        : mean_{i!=j} cos(video_i, video_j)
- modality_gap            : ||mean(text) - mean(video)||_2       — the Liang et al. centroid gap
- text_vs_joint_cos       : mean_i cos(text_i, joint_i)          — deployment-relevant: text query
                                                                   vs the joint doc it should match
- video_vs_joint_cos      : mean_i cos(video_i, joint_i)

Usage:
    python eval/modality_gap.py [--config config/config_norerank.yaml] [--limit N]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.services import get_embedder


def norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def mean_offdiag_cos(A: np.ndarray) -> float:
    S = A @ A.T
    n = S.shape[0]
    return float((S.sum() - np.trace(S)) / (n * (n - 1)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Modality-gap analysis for a VL embedder")
    ap.add_argument("--config", default="config/config_norerank.yaml")
    ap.add_argument("--folder", default="data/scenarios")
    ap.add_argument("--limit", type=int, default=None, help="Sample N scenarios (default: all)")
    args = ap.parse_args()

    cfg = get_config(args.config)
    kw = {}
    if getattr(cfg.embedding, "model_path", None):
        kw["model_path"] = cfg.embedding.model_path
    emb = get_embedder(provider=cfg.embedding.provider, model_name=cfg.embedding.model_name, **kw)

    subs = sorted(d for d in Path(args.folder).iterdir() if d.is_dir())
    if args.limit:
        subs = subs[: args.limit]

    T, V, M = [], [], []
    used = []
    for i, s in enumerate(subs):
        desc = s / "new_description.txt"
        vid = s / "BEV.mp4"
        if not (desc.is_file() and vid.is_file()):
            continue
        text = desc.read_text(encoding="utf-8").strip()
        vp = str(vid.resolve())
        try:
            T.append(emb.encode([{"text": text}])[0])
            V.append(emb.encode([{"video": vp}])[0])
            M.append(emb.encode([{"text": text, "video": vp}])[0])
            used.append(s.name)
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(subs)} embedded", flush=True)
        except Exception as e:
            print(f"  skip {s.name}: {e}", flush=True)

    T, V, M = norm(np.array(T)), norm(np.array(V)), norm(np.array(M))
    matched = float(np.mean(np.sum(T * V, axis=1)))
    text_joint = float(np.mean(np.sum(T * M, axis=1)))
    video_joint = float(np.mean(np.sum(V * M, axis=1)))
    gap = float(np.linalg.norm(T.mean(0) - V.mean(0)))

    result = {
        "embedder": cfg.embedding.model_name,
        "n_scenarios": len(used),
        "matched_cross_modal_cos": round(matched, 4),
        "within_text_cos": round(mean_offdiag_cos(T), 4),
        "within_video_cos": round(mean_offdiag_cos(V), 4),
        "modality_gap_centroid_l2": round(gap, 4),
        "text_vs_joint_cos": round(text_joint, 4),
        "video_vs_joint_cos": round(video_joint, 4),
    }
    out = Path("eval/results/rag/summaries") / f"modality_gap_{cfg.embedding.model_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"MODALITY_GAP_DONE -> {out}")


if __name__ == "__main__":
    main()
