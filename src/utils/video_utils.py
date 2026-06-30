"""Video loading, frame extraction, and timestamp alignment utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import numpy as np

from .config_loader import load_config
from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class VideoMeta:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration_sec: float


def _default_macro_config() -> tuple[int, int]:
    macro = load_config("features")["macro"]
    return int(macro["clip_frames"]), int(macro["clip_stride"])


def _default_micro_size() -> int:
    return int(load_config("features")["micro"]["input_size"])


def get_video_meta(path: str | Path) -> VideoMeta:
    """Read basic video metadata using OpenCV."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()

    if fps <= 0:
        fps = 30.0
        logger.warning("Invalid FPS for %s, falling back to 30.0", path)

    duration_sec = frame_count / fps if frame_count > 0 else 0.0
    return VideoMeta(
        path=str(path),
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        duration_sec=duration_sec,
    )


def _open_capture(path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    return capture


def _frame_index_from_time(time_sec: float, fps: float, max_index: int) -> int:
    index = int(round(time_sec * fps))
    return int(max(0, min(index, max_index)))


def _process_frame(
    frame_bgr: np.ndarray,
    *,
    return_gray: bool,
    target_size: tuple[int, int] | None,
) -> np.ndarray:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if return_gray:
        frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        if target_size is not None:
            frame_rgb = cv2.resize(frame_rgb, target_size, interpolation=cv2.INTER_AREA)
        return frame_rgb

    if target_size is not None:
        frame_rgb = cv2.resize(frame_rgb, target_size, interpolation=cv2.INTER_LINEAR)
    return frame_rgb


def read_frame(
    path: str | Path,
    frame_index: int,
    *,
    return_gray: bool = False,
    target_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Read a single frame by index and return RGB or grayscale numpy array."""
    path = Path(path)
    meta = get_video_meta(path)
    index = max(0, min(frame_index, max(meta.frame_count - 1, 0)))

    capture = _open_capture(path)
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame {index} from {path}")
        return _process_frame(
            frame,
            return_gray=return_gray,
            target_size=target_size,
        )
    finally:
        capture.release()


def extract_frames_at_times(
    video_path: str | Path,
    times_sec: Sequence[float],
    *,
    return_gray: bool = False,
    target_size: tuple[int, int] | None = None,
) -> list[tuple[np.ndarray, float]]:
    """Extract frames at given timestamps in seconds."""
    video_path = Path(video_path)
    meta = get_video_meta(video_path)
    max_index = max(meta.frame_count - 1, 0)

    capture = _open_capture(video_path)
    results: list[tuple[np.ndarray, float]] = []
    try:
        for time_sec in times_sec:
            index = _frame_index_from_time(time_sec, meta.fps, max_index)
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(
                    f"Failed to read frame at {time_sec:.3f}s (index {index}) "
                    f"from {video_path}"
                )
            actual_time = index / meta.fps
            processed = _process_frame(
                frame,
                return_gray=return_gray,
                target_size=target_size,
            )
            results.append((processed, actual_time))
    finally:
        capture.release()

    return results


def sample_frame_indices(
    fps: float,
    duration_sec: float,
    *,
    clip_frames: int | None = None,
    stride: int | None = None,
    start_time_sec: float = 0.0,
) -> list[int]:
    """Return consecutive frame indices for one clip (VideoMAE-style input)."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    if duration_sec < 0:
        raise ValueError("duration_sec must be non-negative")

    default_clip, default_stride = _default_macro_config()
    clip_frames = clip_frames if clip_frames is not None else default_clip
    _ = stride if stride is not None else default_stride  # reserved for sliding windows

    max_frame = max(int(duration_sec * fps) - 1, 0)
    start_index = _frame_index_from_time(start_time_sec, fps, max_frame)
    return [min(start_index + offset, max_frame) for offset in range(clip_frames)]


def extract_frame_clip(
    video_path: str | Path,
    start_frame: int,
    num_frames: int,
    *,
    return_gray: bool = False,
    target_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Extract consecutive frames and return array with shape (T, H, W) or (T, H, W, C)."""
    video_path = Path(video_path)
    meta = get_video_meta(video_path)
    max_index = max(meta.frame_count - 1, 0)
    start = max(0, min(start_frame, max_index))

    capture = _open_capture(video_path)
    frames: list[np.ndarray] = []
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        for offset in range(num_frames):
            index = min(start + offset, max_index)
            if offset > 0:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            frames.append(
                _process_frame(
                    frame,
                    return_gray=return_gray,
                    target_size=target_size,
                )
            )
    finally:
        capture.release()

    if not frames:
        raise RuntimeError(f"No frames extracted from {video_path} starting at {start_frame}")

    return np.stack(frames, axis=0)


def iter_frames(
    path: str | Path,
    *,
    step: int = 1,
    return_gray: bool = False,
    target_size: tuple[int, int] | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Iterate over video frames, yielding ``(frame_index, frame_array)``."""
    if step <= 0:
        raise ValueError("step must be positive")

    path = Path(path)
    capture = _open_capture(path)
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if index % step == 0:
                yield index, _process_frame(
                    frame,
                    return_gray=return_gray,
                    target_size=target_size,
                )
            index += 1
    finally:
        capture.release()


def default_micro_frame_size() -> tuple[int, int]:
    """Return default square size tuple for micro-expression grayscale frames."""
    size = _default_micro_size()
    return (size, size)
