"""Issue #10 implementation for macro-expression feature extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

from src.core.context import DataContext
from src.core.interfaces import FeatureExtractor
from src.core.types import FeatureDict
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.video_utils import extract_frame_clip, get_video_meta

logger = get_logger(__name__)
FeatureList = list[tuple[np.ndarray, float]]
FrameAligner = Callable[[np.ndarray], np.ndarray]
ClipEncoder = Callable[[Sequence[np.ndarray]], np.ndarray]

# Shared VideoMAE weights keyed by model name (avoid reload per pipeline run).
_VIDEOMAE_CACHE: dict[str, tuple[Any, Any]] = {}


class MediaPipeFaceAligner:
    """Align one RGB face from its 468 MediaPipe landmarks."""

    LEFT_EYE = (33, 133)
    RIGHT_EYE = (362, 263)

    def __init__(self, size: int = 224, margin: float = 0.2) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("mediapipe is required for face alignment") from exc
        self.size = int(size)
        self.margin = float(margin)
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=False
        )

    def __call__(self, frame_rgb: np.ndarray) -> np.ndarray:
        result = self._mesh.process(frame_rgb)
        if not result.multi_face_landmarks:
            return _center_crop(frame_rgb, self.size)
        height, width = frame_rgb.shape[:2]
        points = np.asarray(
            [(p.x * width, p.y * height) for p in result.multi_face_landmarks[0].landmark[:468]],
            dtype=np.float32,
        )
        left_eye = points[list(self.LEFT_EYE)].mean(axis=0)
        right_eye = points[list(self.RIGHT_EYE)].mean(axis=0)
        angle = float(np.degrees(np.arctan2(
            right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]
        )))
        matrix = cv2.getRotationMatrix2D(
            tuple(points.mean(axis=0).astype(float)), angle, 1.0
        )
        aligned = cv2.warpAffine(
            frame_rgb, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE
        )
        rotated = cv2.transform(points[None, ...], matrix)[0]
        return _crop_landmark_roi(aligned, rotated, self.size, self.margin)

    def close(self) -> None:
        self._mesh.close()


class MacroExtractor(FeatureExtractor):
    """Extract second-aligned 512-D macro-expression features."""

    modality = "macro"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        aligner: FrameAligner | None = None,
        encoder: ClipEncoder | None = None,
        processor: Any | None = None,
        model: Any | None = None,
    ) -> None:
        cfg = dict(load_config("features")["macro"])
        if config is not None:
            cfg.update(config)
        self.config = cfg
        self.roi_size = int(cfg.get("roi_size", 224))
        self.face_margin = float(cfg.get("face_margin", 0.2))
        self.clip_frames = int(cfg.get("clip_frames", 16))
        self.clip_stride = int(cfg.get("clip_stride", 8))
        self.embedding_dim = int(cfg.get("embedding_dim", 512))
        self.backend = str(cfg.get("backend", "videomae")).lower()
        self.align_faces = bool(cfg.get("align_faces", True))
        self.fallback_to_simple = bool(cfg.get("fallback_to_simple", True))
        if min(self.roi_size, self.clip_frames, self.clip_stride, self.embedding_dim) <= 0:
            raise ValueError(
                "roi_size, clip_frames, clip_stride and embedding_dim must be positive"
            )
        if self.backend not in {"videomae", "simple"}:
            raise ValueError("macro backend must be 'videomae' or 'simple'")
        self._aligner = aligner
        self._encoder = encoder
        self._processor = processor
        self._model = model
        self._last_features: FeatureList = []

    def extract(self, context: DataContext) -> FeatureDict:
        if context.input_type == "image":
            path = context.raw_data.get("image_path")
            if not path:
                raise ValueError("DataContext.raw_data['image_path'] is required")
            features = self.extract_image(path)
        elif context.input_type == "video":
            path = context.raw_data.get("video_path")
            if not path:
                raise ValueError("DataContext.raw_data['video_path'] is required")
            features = self.extract_video(path)
        else:
            raise ValueError("MacroExtractor supports only video and image input profiles")
        self._last_features = features
        return {self.modality: features}

    def extract_raw_visual(self, context: DataContext) -> dict[str, Any]:
        return {self.modality: self._last_features}

    def extract_video(self, path: str | Path) -> FeatureList:
        meta = get_video_meta(path)
        if meta.frame_count <= 0:
            return []
        last_start = max(meta.frame_count - self.clip_frames, 0)
        starts = list(range(0, last_start + 1, self.clip_stride))
        if last_start not in starts:
            starts.append(last_start)
        clips: FeatureList = []
        for start in starts:
            frames = _pad_clip(
                extract_frame_clip(path, start, self.clip_frames), self.clip_frames
            )
            aligned = [self._get_aligner()(frame) for frame in frames]
            midpoint = min(start + (self.clip_frames - 1) / 2, meta.frame_count - 1)
            clips.append((self._encode(aligned), midpoint / meta.fps))
        return _align_to_seconds(clips, meta.duration_sec)

    def extract_image(self, path: str | Path) -> FeatureList:
        image_path = Path(path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        frame_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        roi = self._get_aligner()(frame_rgb)
        return [(self._encode([roi.copy() for _ in range(self.clip_frames)]), 0.0)]

    def _get_aligner(self) -> FrameAligner:
        if self._aligner is not None:
            return self._aligner
        if not self.align_faces:
            self._aligner = lambda frame: _center_crop(frame, self.roi_size)
            return self._aligner
        try:
            self._aligner = MediaPipeFaceAligner(self.roi_size, self.face_margin)
        except (RuntimeError, AttributeError):
            if not self.fallback_to_simple:
                raise
            logger.warning("MediaPipe unavailable; falling back to centered crops")
            self._aligner = lambda frame: _center_crop(frame, self.roi_size)
        return self._aligner

    def _encode(self, clip: Sequence[np.ndarray]) -> np.ndarray:
        if self._encoder is not None:
            return _resize_and_normalize(self._encoder(clip), self.embedding_dim)
        if self.backend == "simple":
            return _simple_embedding(clip, self.embedding_dim)
        try:
            return self._encode_videomae(clip)
        except (ImportError, OSError, RuntimeError) as exc:
            if not self.fallback_to_simple:
                raise
            logger.warning("VideoMAE unavailable (%s); using simple encoder", exc)
            self.backend = "simple"
            return _simple_embedding(clip, self.embedding_dim)

    def _encode_videomae(self, clip: Sequence[np.ndarray]) -> np.ndarray:
        import torch
        from transformers import VideoMAEImageProcessor, VideoMAEModel

        model_name = str(
            self.config.get(
                "videomae_model", "MCG-NJU/videomae-base-finetuned-kinetics"
            )
        )
        if self._processor is None or self._model is None:
            cached = _VIDEOMAE_CACHE.get(model_name)
            if cached is not None:
                self._processor, self._model = cached
            else:
                processor = VideoMAEImageProcessor.from_pretrained(model_name)
                model = VideoMAEModel.from_pretrained(model_name).eval()
                _VIDEOMAE_CACHE[model_name] = (processor, model)
                self._processor, self._model = processor, model
        device = str(self.config.get("device", "auto"))
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(device)
        inputs = self._processor(list(clip), return_tensors="pt")
        inputs = {name: value.to(device) for name, value in inputs.items()}
        with torch.inference_mode():
            vector = self._model(**inputs).last_hidden_state.mean(dim=1)[0]
        return _resize_and_normalize(vector.detach().cpu().numpy(), self.embedding_dim)


def _center_crop(frame: np.ndarray, size: int) -> np.ndarray:
    height, width = frame.shape[:2]
    side = min(height, width)
    x0, y0 = (width - side) // 2, (height - side) // 2
    return cv2.resize(frame[y0:y0 + side, x0:x0 + side], (size, size))


def _crop_landmark_roi(
    frame: np.ndarray, points: np.ndarray, size: int, margin: float
) -> np.ndarray:
    height, width = frame.shape[:2]
    low, high = points.min(axis=0), points.max(axis=0)
    center = (low + high) / 2
    side = float(max(high - low)) * (1.0 + margin)
    x0, y0 = np.floor(center - side / 2).astype(int)
    x1, y1 = np.ceil(center + side / 2).astype(int)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, width), min(y1, height)
    if x1 <= x0 or y1 <= y0:
        return _center_crop(frame, size)
    return cv2.resize(frame[y0:y1, x0:x1], (size, size))


def _pad_clip(frames: np.ndarray, target_length: int) -> np.ndarray:
    if len(frames) == 0:
        raise ValueError("Cannot pad an empty clip")
    if len(frames) >= target_length:
        return frames[:target_length]
    padding = np.repeat(frames[-1][None, ...], target_length - len(frames), axis=0)
    return np.concatenate((frames, padding), axis=0)


def _resize_and_normalize(vector: Any, size: int) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return np.zeros(size, dtype=np.float32)
    if values.size != size:
        values = np.interp(
            np.linspace(0, values.size - 1, size), np.arange(values.size), values
        ).astype(np.float32)
    norm = float(np.linalg.norm(values))
    return (values / norm if norm else values).astype(np.float32, copy=False)


def _simple_embedding(clip: Sequence[np.ndarray], size: int) -> np.ndarray:
    gray = np.stack([
        cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (16, 16))
        for frame in clip
    ]).astype(np.float32) / 255.0
    motion = np.abs(np.diff(gray, axis=0)).mean(axis=0) if len(gray) > 1 else np.zeros_like(gray[0])
    return _resize_and_normalize(
        np.concatenate((gray.mean(axis=0).ravel(), motion.ravel())), size
    )


def _align_to_seconds(clips: FeatureList, duration_sec: float) -> FeatureList:
    if not clips:
        return []
    times = np.asarray([timestamp for _, timestamp in clips], dtype=np.float64)
    matrix = np.stack([feature for feature, _ in clips])
    output: FeatureList = []
    for target in np.arange(max(int(np.ceil(duration_sec)), 1), dtype=np.float64):
        right = int(np.searchsorted(times, target))
        if right == 0 or len(clips) == 1:
            vector = matrix[0]
        elif right == len(clips):
            vector = matrix[-1]
        else:
            left = right - 1
            span = times[right] - times[left]
            weight = 0.0 if span <= 0 else (target - times[left]) / span
            vector = matrix[left] * (1.0 - weight) + matrix[right] * weight
        output.append(
            (_resize_and_normalize(vector, matrix.shape[1]), float(target))
        )
    return output


# Compatibility with the name used by the earlier prototype.
MacroExpressionExtractor = MacroExtractor

__all__ = [
    "MacroExtractor", "MacroExpressionExtractor", "MediaPipeFaceAligner"
]
