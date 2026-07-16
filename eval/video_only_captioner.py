"""
Generate PURE-VIDEO captions for each scenario: BEV.mp4 -> qwen3.6-plus -> video_only_description.txt

Unlike text_only_rag.py (which feeds the VLM both the video AND description.txt), this captioner
sees ONLY the video — required for the honest "video-only input via caption bridge" condition
(video -> MLLM caption -> text-only embedder).

Endpoint: Qwen token plan, OpenAI-compatible (QWEN_API_KEY in .env).
Video is sent inline as a base64 data URL.

Usage:
    python eval/video_only_captioner.py [--folder data/scenarios] [--limit N] [--force]
"""

import argparse
import base64
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("video_only_captioner")

BASE_URL = "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen3.6-plus"
OUTPUT_FILENAME = "video_only_description.txt"

# Mirrors the output format of text_query_description.txt (flattened-DSL register) so the
# caption lives in the same textual style as the indexed new_description.txt corpus.
SYSTEM_PROMPT = """You are an expert autonomous driving scenario analyzer. You are given ONLY a bird's-eye-view (BEV) video of a driving scenario. Describe the scenario in the following exact plain-text format (no markdown, no JSON, no extra sections):

Scenario: [One sentence summarizing the scenario.]
Ego: [Ego object type]: [A short description of the ego behavior.]
Adversarial Entities: [Object type]: [A short description of the behavior.] (one per entity, separated by " | ")
Spatial Relation: [A single short sentence defining the road type and the relative positioning of all entities.]
Requirements and Restrictions: [Initial distance constraints and termination conditions, if visible.]

Rules:
- The vehicle nearest the center of the frame at the start is the Ego; all other dynamic entities are Adversarial.
- Track Initial, Midpoint, and Ending phases of the video to deduce behaviors (accelerating, yielding, stopping, turning).
- Subject-first descriptions ("The ego vehicle travels...").
- Allowed Ego objects: Car, Motorcycle, Truck.
- Allowed Adversarial objects: Car, NPCCar, Bicycle, Motorcycle, Truck, Pedestrian, Trash, Garbage, Container, CreasedBox, Case, Box, Gnome.
- No specific numbers, distances, speeds, or durations (say "a certain distance").
- Do NOT include a reasoning chain or any commentary — output ONLY the five sections above."""


def _frames_content(video_path: Path, num_frames: int = 8) -> list:
    """Sample frames and send them as an image-list 'video' — fallback for clips the API
    rejects as too short (no min-duration requirement on frame lists)."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idxs = [int(i * (total - 1) / max(num_frames - 1, 1)) for i in range(min(num_frames, total))]
    urls = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            urls.append(f"data:image/jpeg;base64,{base64.b64encode(buf.tobytes()).decode()}")
    cap.release()
    if len(urls) < 2:
        raise RuntimeError(f"could not extract enough frames from {video_path}")
    return [
        {"type": "video", "video": urls},
        {"type": "text", "text": "Describe this driving scenario in the required format."},
    ]


def caption_video(client: OpenAI, video_path: Path, max_retries: int = 3) -> str:
    video_b64 = base64.b64encode(video_path.read_bytes()).decode()
    content = [
        {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}},
        {"type": "text", "text": "Describe this driving scenario in the required format."},
    ]
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
            )
            text = (resp.choices[0].message.content or "").strip()
            if not text:
                raise RuntimeError("empty completion")
            return text
        except Exception as exc:
            last_exc = exc
            if "too short" in str(exc):
                logger.info("%s rejected as too short — retrying with sampled-frame list", video_path.name)
                content = _frames_content(video_path)
                continue
            wait = 5 * attempt
            logger.warning("attempt %d/%d failed (%s); retrying in %ds", attempt, max_retries, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"captioning failed after {max_retries} attempts: {last_exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pure-video captions via qwen3.6-plus")
    parser.add_argument("--folder", default="data/scenarios")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N scenarios (smoke test)")
    parser.add_argument("--force", action="store_true", help="Regenerate even if output exists")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("QWEN_API_KEY")
    if not api_key:
        raise SystemExit("QWEN_API_KEY not set (check .env)")
    client = OpenAI(base_url=BASE_URL, api_key=api_key, timeout=180)

    root = Path(args.folder)
    subfolders = sorted(d for d in root.iterdir() if d.is_dir())
    if args.limit:
        subfolders = subfolders[: args.limit]

    processed, skipped, failed = [], [], []
    for i, sub in enumerate(subfolders):
        out = sub / OUTPUT_FILENAME
        if out.is_file() and not args.force:
            skipped.append(sub.name)
            continue
        bev = sub / "BEV.mp4"
        if not bev.is_file():
            failed.append(sub.name)
            logger.warning("[%d/%d] missing BEV.mp4: %s", i + 1, len(subfolders), sub.name)
            continue
        try:
            text = caption_video(client, bev)
            out.write_text(text, encoding="utf-8")
            processed.append(sub.name)
            logger.info("[%d/%d] captioned %s (%d chars)", i + 1, len(subfolders), sub.name, len(text))
        except Exception as exc:
            failed.append(sub.name)
            logger.error("[%d/%d] FAILED %s: %s", i + 1, len(subfolders), sub.name, exc)

    logger.info("done: %d processed, %d skipped, %d failed", len(processed), len(skipped), len(failed))
    if failed:
        logger.error("failed: %s", failed)
    print(f"VCAPTION_DONE processed={len(processed)} skipped={len(skipped)} failed={len(failed)}")


if __name__ == "__main__":
    main()
