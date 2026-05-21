"""Video frame sampling helpers for OCR ingestion."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

try:
    import cv2

    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    _CV2_AVAILABLE = False


DEFAULT_VIDEO_SAMPLE_SECONDS = 1.0
DEFAULT_VIDEO_MAX_FRAMES = 300


def normalize_video_sampling(
    sample_seconds: float = DEFAULT_VIDEO_SAMPLE_SECONDS,
    max_frames: int = DEFAULT_VIDEO_MAX_FRAMES,
) -> tuple[float, int]:
    """Clamp sampling inputs into conservative operational bounds."""
    try:
        normalized_seconds = float(sample_seconds)
    except (TypeError, ValueError):
        normalized_seconds = DEFAULT_VIDEO_SAMPLE_SECONDS

    if not math.isfinite(normalized_seconds):
        normalized_seconds = DEFAULT_VIDEO_SAMPLE_SECONDS
    normalized_seconds = min(max(normalized_seconds, 0.1), 3600.0)

    try:
        normalized_max_frames = int(max_frames)
    except (TypeError, ValueError):
        normalized_max_frames = DEFAULT_VIDEO_MAX_FRAMES

    normalized_max_frames = min(max(normalized_max_frames, 1), 10_000)
    return normalized_seconds, normalized_max_frames


def _normalize_fps(fps: float) -> float:
    """Convert unreliable FPS metadata into a safe positive value."""
    try:
        normalized = float(fps)
    except (TypeError, ValueError):
        normalized = 0.0
    if not math.isfinite(normalized) or normalized <= 0.0:
        return 1.0
    return normalized


def _downsample_indices(indices: list[int], max_frames: int) -> list[int]:
    """Reduce a frame plan deterministically while keeping endpoints."""
    if len(indices) <= max_frames:
        return list(indices)
    if max_frames == 1:
        return [indices[0]]

    last_position = len(indices) - 1
    sampled: list[int] = []
    for slot in range(max_frames):
        position = round(slot * last_position / (max_frames - 1))
        frame_index = indices[int(position)]
        if not sampled or frame_index != sampled[-1]:
            sampled.append(frame_index)
    return sampled


def build_video_frame_plan(
    frame_count: int,
    fps: float,
    sample_seconds: float = DEFAULT_VIDEO_SAMPLE_SECONDS,
    max_frames: int = DEFAULT_VIDEO_MAX_FRAMES,
) -> list[int]:
    """Build the sampled zero-based frame indices for a video source."""
    if frame_count <= 0:
        return []

    normalized_seconds, normalized_max_frames = normalize_video_sampling(
        sample_seconds,
        max_frames,
    )
    stride_frames = max(1, int(round(normalized_seconds * _normalize_fps(fps))))

    indices = list(range(0, frame_count, stride_frames))
    last_frame = frame_count - 1
    if indices[-1] != last_frame:
        indices.append(last_frame)
    return _downsample_indices(indices, normalized_max_frames)


def _open_video_capture(path: str):
    """Open a video capture handle or raise a descriptive error."""
    if not _CV2_AVAILABLE or cv2 is None:
        raise RuntimeError("OpenCV video support not available")

    capture = cv2.VideoCapture(path)
    if capture is None or not capture.isOpened():
        if capture is not None:
            capture.release()
        raise RuntimeError(f"Unable to open video source: {path}")
    return capture


def _count_video_frames(path: str) -> int:
    """Count frames by decoding when container metadata is missing."""
    capture = _open_video_capture(path)
    count = 0
    try:
        while True:
            ok, _frame = capture.read()
            if not ok:
                break
            count += 1
    finally:
        capture.release()
    return count


def probe_video_metadata(path: str) -> tuple[int, float]:
    """Return `(frame_count, fps)` for a video source."""
    capture = _open_video_capture(path)
    try:
        frame_count = int(round(float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        capture.release()

    if frame_count <= 0:
        frame_count = _count_video_frames(path)
    return frame_count, fps


def get_video_page_count(
    path: str,
    sample_seconds: float = DEFAULT_VIDEO_SAMPLE_SECONDS,
    max_frames: int = DEFAULT_VIDEO_MAX_FRAMES,
) -> int:
    """Return the number of sampled OCR pages represented by a video."""
    frame_count, fps = probe_video_metadata(path)
    return len(build_video_frame_plan(frame_count, fps, sample_seconds, max_frames))


def _frame_to_pil(frame: np.ndarray) -> Image.Image:
    """Convert an OpenCV frame array into a PIL RGB image."""
    if frame.ndim == 2:
        return Image.fromarray(frame).convert("RGB")
    if frame.ndim != 3:
        raise RuntimeError(f"Unsupported video frame shape: {frame.shape!r}")

    channels = frame.shape[2]
    if channels == 3:
        rgb = frame[:, :, ::-1]
    elif channels == 4:
        rgb = frame[:, :, [2, 1, 0, 3]]
    else:
        raise RuntimeError(f"Unsupported video frame channels: {channels}")
    return Image.fromarray(rgb).convert("RGB")


def iter_video_frames(
    path: str,
    start: int,
    end: int,
    sample_seconds: float = DEFAULT_VIDEO_SAMPLE_SECONDS,
    max_frames: int = DEFAULT_VIDEO_MAX_FRAMES,
):
    """Yield sampled video frames as PIL RGB images for a 1-based page range."""
    frame_count, fps = probe_video_metadata(path)
    plan = build_video_frame_plan(frame_count, fps, sample_seconds, max_frames)
    if not plan:
        return

    lower = max(1, int(start))
    upper = min(int(end), len(plan))
    if lower > upper:
        return

    capture = _open_video_capture(path)
    try:
        for page_num in range(lower, upper + 1):
            frame_index = plan[page_num - 1]
            capture.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
            ok, frame = capture.read()
            if not ok or frame is None or getattr(frame, "size", 0) == 0:
                raise RuntimeError(f"Failed to decode video frame {frame_index} from {path}")
            yield _frame_to_pil(frame)
    finally:
        capture.release()
