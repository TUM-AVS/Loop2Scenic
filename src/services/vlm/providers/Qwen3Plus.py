"""Qwen VLM via the same OpenAI-compatible MaaS endpoint as the LLM / video captioner.

No WORKSPACE_ID required. Media is NOT uploaded permanently:
- Images: inlined as base64 data URLs (downscaled if large).
- Videos: sampled JPEG frames sent as image_url parts (not video modality),
  which avoids both 413 body limits and "video too short" 400s.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

from src.services.vlm.base import BaseVLMModel

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.6-plus"

_MAX_INLINE_BYTES = 2 * 1024 * 1024  # 2 MiB
# Prefer temporal coverage over per-frame fidelity (critic often sends 2 videos).
_VIDEO_NUM_FRAMES = 8
_VIDEO_JPEG_QUALITY = 40
_IMAGE_MAX_SIDE = 480


class Qwen3Plus(BaseVLMModel):
    """Alibaba Qwen multimodal models over the OpenAI-compatible chat API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        model: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0,
        max_tokens: int = 8192,
        **kwargs,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("OpenAI package not installed. Install with: pip install openai") from exc

        if not api_key:
            from dotenv import load_dotenv

            load_dotenv()
            api_key = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("QWEN_API_KEY (or DASHSCOPE_API_KEY) environment variable is not set.")

        self._model_name = model or model_name or DEFAULT_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens
        timeout = kwargs.pop("timeout", 180)
        kwargs.pop("model_path", None)
        kwargs.pop("default_instruction", None)
        kwargs.pop("device", None)

        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        # token -> OpenAI content part(s); lets agents pass Part.from_uri(token) without huge bodies
        self._media_registry: dict[str, list[dict[str, Any]]] = {}
        self._metrics = {
            "calls": 0,
            "response_time_ms": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        logger.info(
            "Initialized Qwen VLM: %s (endpoint=%s, temperature=%s, max_tokens=%s)",
            self._model_name,
            base_url,
            temperature,
            max_tokens,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    def load_media(self, file_path: str) -> SimpleNamespace:
        """Prepare local media for the API.

        Videos → sampled JPEG frames as ``image_url`` parts (never full MP4 / video
        modality). Returns ``.uri`` / ``.mime_type`` for Gemini-style ``Part.from_uri``.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Local file not found: {file_path}")

        mime_type, _ = mimetypes.guess_type(str(path))
        if mime_type is None:
            mime_type = {
                ".mp4": "video/mp4",
                ".mov": "video/quicktime",
                ".avi": "video/x-msvideo",
                ".webm": "video/webm",
                ".mkv": "video/x-matroska",
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
                ".gif": "image/gif",
            }.get(path.suffix.lower(), "application/octet-stream")

        token = f"qwen-media://{uuid.uuid4().hex}"
        if mime_type.startswith("video/") or path.suffix.lower() in {".mp4", ".mov", ".avi", ".webm", ".mkv"}:
            parts = self._video_as_image_parts(path)
            size_mb = path.stat().st_size / (1024 * 1024)
            n_imgs = sum(1 for p in parts if p.get("type") == "image_url")
            logger.info(
                "Qwen VLM: video %s (%.1f MiB) → %d frame image_url(s)",
                path.name,
                size_mb,
                n_imgs,
            )
        else:
            data_url = self._image_data_url(path, mime_type)
            parts = [self._openai_image_part(data_url)]

        self._media_registry[token] = parts
        return SimpleNamespace(uri=token, mime_type=mime_type, name=str(path))

    def chat(
        self,
        text: Optional[str] = None,
        image: Optional[str] = None,
        video: Optional[str] = None,
        instruction: Optional[str] = None,
        **kwargs,
    ) -> str:
        content: list[dict[str, Any]] = []
        if image:
            content.extend(self._media_registry[self.load_media(image).uri])
        if video:
            content.extend(self._media_registry[self.load_media(video).uri])
        if text:
            content.append({"type": "text", "text": text})
        if not content:
            content.append({"type": "text", "text": "Please describe what you see."})

        return self.chat_with_content(
            contents=[{"role": "user", "content": content}],
            system_instruction=instruction,
            **kwargs,
        )

    def chat_with_content(
        self,
        contents: Any,
        system_instruction: Optional[str] = None,
        **kwargs,
    ) -> str:
        messages = self._to_openai_messages(contents, system_instruction)
        messages = self._demote_video_modality(messages)
        temperature = kwargs.pop("temperature", self.temperature)
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)
        call_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ("temperature", "max_tokens", "timeout", "response_format")
        }

        start = time.perf_counter()
        response = None
        last_exc: Exception | None = None
        for attempt in range(1, 4):
            try:
                response = self.client.chat.completions.create(
                    model=self._model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    # extra_body={"enable_thinking": True},
                    **call_kwargs,
                )
                break
            except Exception as exc:
                last_exc = exc
                kind = self._classify_api_error(exc)
                logger.warning(
                    "Qwen VLM attempt %d/3 failed (%s): %s",
                    attempt,
                    kind,
                    str(exc)[:200],
                )
                if kind == "too_short":
                    messages = self._demote_video_modality(messages)
                    continue
                if kind in {"too_large", "input_length"}:
                    messages = self._shrink_visual_payload(messages, aggressiveness=attempt)
                    continue
                raise
        if response is None:
            raise last_exc  # type: ignore[misc]

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)
        self._metrics["calls"] += 1
        self._metrics["response_time_ms"] += elapsed_ms
        self._metrics["prompt_tokens"] += prompt_tokens
        self._metrics["completion_tokens"] += completion_tokens
        self._metrics["total_tokens"] += total_tokens

        text = (response.choices[0].message.content or "").strip()
        if not text:
            raise ValueError(f"Empty response from Qwen VLM ({self._model_name})")
        return text

    def get_metrics_snapshot(self) -> dict:
        return dict(self._metrics)

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _classify_api_error(exc: Exception) -> str:
        msg = str(exc).lower()
        if "413" in msg or "requesttoolarge" in msg or "body size exceeds" in msg:
            return "too_large"
        if "too short" in msg:
            return "too_short"
        if "991808" in msg or "input length" in msg or "range of input length" in msg:
            return "input_length"
        return "other"

    @staticmethod
    def _openai_image_part(url: str) -> dict[str, Any]:
        return {"type": "image_url", "image_url": {"url": url}}

    def _video_as_image_parts(
        self,
        video_path: Path,
        num_frames: int = _VIDEO_NUM_FRAMES,
        max_side: int = _IMAGE_MAX_SIDE,
        jpeg_quality: int = _VIDEO_JPEG_QUALITY,
    ) -> list[dict[str, Any]]:
        """Sample frames and emit image_url parts (avoids video modality checks)."""
        frame_urls = self._sample_video_frames(
            video_path,
            num_frames=num_frames,
            max_side=max_side,
            jpeg_quality=jpeg_quality,
        )
        return (
            [{"type": "text", "text": f"[Video frames from {video_path.name}]"}]
            + [self._openai_image_part(u) for u in frame_urls]
        )

    def _sample_video_frames(
        self,
        video_path: Path,
        num_frames: int = _VIDEO_NUM_FRAMES,
        max_side: int = _IMAGE_MAX_SIDE,
        jpeg_quality: int = _VIDEO_JPEG_QUALITY,
    ) -> list[str]:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        n = max(1, min(num_frames, total))
        if n == 1:
            idxs = [0]
        else:
            idxs = [int(i * (total - 1) / (n - 1)) for i in range(n)]
        urls: list[str] = []
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            scale = min(1.0, max_side / max(h, w))
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            if ok:
                urls.append(f"data:image/jpeg;base64,{base64.b64encode(buf.tobytes()).decode()}")
        cap.release()
        if not urls:
            raise RuntimeError(f"Could not extract frames from {video_path}")
        return urls

    def _image_data_url(self, path: Path, mime_type: str) -> str:
        raw = path.read_bytes()
        if len(raw) <= _MAX_INLINE_BYTES and path.suffix.lower() in {".jpg", ".jpeg", ".webp"}:
            return f"data:{mime_type};base64,{base64.b64encode(raw).decode()}"

        try:
            import cv2
            import numpy as np

            arr = np.frombuffer(raw, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return f"data:{mime_type};base64,{base64.b64encode(raw).decode()}"
            h, w = img.shape[:2]
            scale = min(1.0, _IMAGE_MAX_SIDE / max(h, w))
            if scale < 1.0:
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, _VIDEO_JPEG_QUALITY])
            if not ok:
                return f"data:{mime_type};base64,{base64.b64encode(raw).decode()}"
            logger.info(
                "Qwen VLM: image %s re-encoded (%d → %d bytes)",
                path.name,
                len(raw),
                len(buf),
            )
            return f"data:image/jpeg;base64,{base64.b64encode(buf.tobytes()).decode()}"
        except Exception as exc:
            logger.warning("Image downscale failed for %s (%s); sending original", path.name, exc)
            return f"data:{mime_type};base64,{base64.b64encode(raw).decode()}"

    def _resolve_registry_uri(self, uri: str) -> list[dict[str, Any]]:
        if uri in self._media_registry:
            return self._media_registry[uri]
        return []

    def _media_part_from_uri(self, uri: str, mime_type: Optional[str] = None) -> list[dict[str, Any]]:
        registered = self._resolve_registry_uri(uri)
        if registered:
            return registered

        mime = (mime_type or "").lower()
        if not mime and uri.startswith("data:"):
            mime = uri[5:].split(";", 1)[0].lower()

        if uri.startswith("data:video") or (
            mime.startswith("video/") and uri.startswith("data:") and len(uri) > _MAX_INLINE_BYTES
        ):
            logger.warning(
                "Dropping oversized inline video data URL (%d chars); use load_media(path)",
                len(uri),
            )
            return [{"type": "text", "text": "[video omitted: payload too large for API]"}]

        if mime.startswith("video/") or uri.lower().endswith((".mp4", ".mov", ".avi", ".webm", ".mkv")):
            path = Path(uri)
            if path.is_file():
                return self._video_as_image_parts(path)
            # Remote / unknown video URL — demote to text to avoid "too short" / 413
            logger.warning("Omitting non-local video URI: %s", uri[:80])
            return [{"type": "text", "text": "[video omitted: non-local URI]"}]

        return [self._openai_image_part(uri)]

    def _convert_part(self, part: Any) -> list[dict[str, Any]]:
        if part is None:
            return []

        if isinstance(part, dict):
            ptype = part.get("type")
            if ptype == "video_url":
                url = (part.get("video_url") or {}).get("url", "")
                if isinstance(url, str) and url.startswith("qwen-media://"):
                    return self._resolve_registry_uri(url)
                if isinstance(url, str) and url.startswith("data:video"):
                    logger.warning("Demoting inline video_url to omission (use load_media)")
                    return [{"type": "text", "text": "[video omitted: use frame sampling]"}]
                if isinstance(url, str) and Path(url).is_file():
                    return self._video_as_image_parts(Path(url))
                return [{"type": "text", "text": "[video omitted]"}]
            if ptype == "video" or "video" in part and ptype is None:
                v = part.get("video")
                if isinstance(v, list):
                    # Frame list already — send as image_urls
                    return (
                        [{"type": "text", "text": "[Video frames]"}]
                        + [self._openai_image_part(u) if isinstance(u, str) else u for u in v]
                    )
                if isinstance(v, str):
                    return self._media_part_from_uri(v, "video/mp4")
            if ptype == "image_url" or ptype == "text":
                return [part]
            if "type" in part:
                return [part]
            if "text" in part and len(part) == 1:
                return [{"type": "text", "text": part["text"]}]
            if "image" in part:
                return [self._openai_image_part(part["image"])]
            if "image_url" in part:
                return [part]
            return []

        if isinstance(part, str):
            if part.startswith("qwen-media://"):
                return self._resolve_registry_uri(part)
            return [{"type": "text", "text": part}]

        uri = getattr(part, "uri", None)
        mime = getattr(part, "mime_type", None)
        if isinstance(uri, str) and uri:
            return self._media_part_from_uri(uri, mime)

        text = getattr(part, "text", None)
        if text:
            return [{"type": "text", "text": text}]

        file_data = getattr(part, "file_data", None)
        if file_data is not None:
            file_uri = getattr(file_data, "file_uri", None) or getattr(file_data, "uri", None)
            file_mime = getattr(file_data, "mime_type", None)
            if file_uri:
                return self._media_part_from_uri(str(file_uri), file_mime)

        inline = getattr(part, "inline_data", None)
        if inline is not None:
            data = getattr(inline, "data", None)
            inline_mime = getattr(inline, "mime_type", None) or "application/octet-stream"
            if data is not None:
                if isinstance(data, bytes):
                    b64 = base64.b64encode(data).decode("utf-8")
                else:
                    b64 = data
                return self._media_part_from_uri(f"data:{inline_mime};base64,{b64}", inline_mime)

        logger.warning("Skipping unsupported VLM content part: %r", type(part))
        return []

    def _to_openai_messages(
        self,
        contents: Any,
        system_instruction: Optional[str],
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})

        if (
            isinstance(contents, list)
            and contents
            and isinstance(contents[0], dict)
            and "role" in contents[0]
        ):
            for msg in contents:
                role = msg.get("role", "user")
                content = msg.get("content")
                if isinstance(content, list):
                    flat: list[dict[str, Any]] = []
                    for item in content:
                        flat.extend(self._convert_part(item))
                    messages.append({"role": role, "content": flat or content})
                else:
                    messages.append(msg)
            return messages

        if isinstance(contents, list):
            flat = []
            for item in contents:
                flat.extend(self._convert_part(item))
            messages.append({"role": "user", "content": flat or [{"type": "text", "text": str(contents)}]})
            return messages

        flat = self._convert_part(contents)
        messages.append({"role": "user", "content": flat or [{"type": "text", "text": str(contents)}]})
        return messages

    def _demote_video_modality(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert any remaining video / video_url parts to image_url or text."""
        out = []
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                out.append(msg)
                continue
            new_content: list[dict[str, Any]] = []
            for part in content:
                if not isinstance(part, dict):
                    new_content.append(part)
                    continue
                if part.get("type") in {"video", "video_url"} or "video" in part:
                    new_content.extend(self._convert_part(part))
                else:
                    new_content.append(part)
            out.append({**msg, "content": new_content})
        return out

    def _shrink_visual_payload(
        self,
        messages: list[dict[str, Any]],
        aggressiveness: int = 1,
    ) -> list[dict[str, Any]]:
        """Drop / thin image_url parts to satisfy body-size and input-length limits."""
        keep_every = 1 + aggressiveness  # attempt1→2, attempt2→3
        out = []
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                out.append(msg)
                continue
            new_content: list[dict[str, Any]] = []
            img_i = 0
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    url = (part.get("image_url") or {}).get("url", "")
                    # Always drop leftover full-video data URLs
                    if isinstance(url, str) and url.startswith("data:video"):
                        continue
                    if img_i % keep_every == 0:
                        new_content.append(part)
                    img_i += 1
                elif isinstance(part, dict) and part.get("type") in {"video", "video_url"}:
                    continue
                else:
                    new_content.append(part)
            # If still many images, keep only the first few
            max_imgs = max(2, 8 // aggressiveness)
            kept_imgs = 0
            trimmed: list[dict[str, Any]] = []
            for part in new_content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    if kept_imgs >= max_imgs:
                        continue
                    kept_imgs += 1
                trimmed.append(part)
            logger.info(
                "Qwen VLM: shrunk visuals (aggressiveness=%d) → %d image_url(s)",
                aggressiveness,
                kept_imgs,
            )
            out.append({**msg, "content": trimmed})
        return out


if __name__ == "__main__":
    vlm = Qwen3Plus(model="qwen3.6-plus")
    response = vlm.chat(
        text="Please describe what happened in the scenario in the image and video.",
        image="/home/dellpro2/chenli/ads-mrag/ads-mrag/data/processed/test_data/testimage.png",
        video="/home/dellpro2/chenli/ads-mrag/ads-mrag/data/processed/test_data/testvideo.mp4",
    )
    print(response)
