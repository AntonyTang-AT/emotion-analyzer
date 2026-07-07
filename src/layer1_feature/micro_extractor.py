"""L1 micro-expression extraction using optical-flow displacement features.

The production FRL-DGT path calls for a trained DGM and landmark GCN. For the
initial project stage we implement the documented lightweight substitute:
TV-L1 optical flow when OpenCV exposes it, with Farneback as a deterministic
fallback. The extractor keeps the public schema stable while avoiding model
downloads during tests and local development.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.core.context import DataContext
from src.core.interfaces import FeatureExtractor
from src.core.types import FeatureDict
from src.layer1_feature.dgm import DisplacementFieldGenerator
from src.layer1_feature.gcn import MultiViewMicroProjector
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class MicroExpressionEvent:
    """A lightweight onset-apex event candidate."""

    onset_index: int
    apex_index: int
    start_time: float
    end_time: float
    onset_frame: np.ndarray
    apex_frame: np.ndarray


class MicroExpressionExtractor(FeatureExtractor):
    """Extract 256-dim micro-expression vectors from video or image inputs."""

    modality = "micro"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        micro_cfg = (config or load_config("features"))["micro"]
        self.input_size = int(micro_cfg.get("input_size", 112))
        self.embedding_dim = int(micro_cfg.get("embedding_dim", 256))
        if self.embedding_dim != 256:
            raise ValueError("MicroExpressionExtractor currently requires 256 dims")

        amplify_cfg = micro_cfg.get("weak_signal_amplify", {})
        self.amplify_enabled = bool(amplify_cfg.get("enabled", True))
        self.amplify_coeff = float(amplify_cfg.get("au_coefficients", 2.5))
        self.preserve_raw_visual = bool(micro_cfg.get("preserve_raw_visual", True))

        self.displacement = DisplacementFieldGenerator(size=self.input_size)
        self.projector = MultiViewMicroProjector(
            content_dim=128,
            relation_dim=128,
        )
        self._raw_visual: list[np.ndarray] = []

    def extract(self, context: DataContext) -> FeatureDict:
        events, fps = self._resolve_events(context)
        results: list[dict[str, Any]] = []
        raw_visual: list[np.ndarray] = []

        for event in events:
            flow = self.displacement.compute(event.onset_frame, event.apex_frame)
            if self.amplify_enabled:
                flow = self._amplify_weak_signal(flow)

            feature = self.projector.project(flow).astype(np.float32)
            if feature.shape != (self.embedding_dim,):
                raise ValueError(
                    f"Expected micro feature dim {self.embedding_dim}, "
                    f"got {feature.shape}"
                )

            raw_visual.append(feature.copy())
            results.append(
                {
                    "micro_feature": feature,
                    "start_time": float(event.start_time),
                    "end_time": float(event.end_time),
                    "onset_frame": int(event.onset_index),
                    "apex_frame": int(event.apex_index),
                    "fps": float(fps),
                }
            )

        self._raw_visual = raw_visual
        return {self.modality: results}

    def extract_raw_visual(self, context: DataContext) -> dict[str, Any]:
        if not self.preserve_raw_visual:
            return {}
        return {self.modality: [item.copy() for item in self._raw_visual]}

    def _resolve_events(self, context: DataContext) -> tuple[list[MicroExpressionEvent], float]:
        if context.input_type == "image":
            image_path = context.raw_data.get("image_path")
            if not image_path:
                raise ValueError("MicroExpressionExtractor requires image_path")
            frame = self._read_image(image_path)
            return [
                MicroExpressionEvent(
                    onset_index=0,
                    apex_index=0,
                    start_time=0.0,
                    end_time=0.0,
                    onset_frame=frame,
                    apex_frame=frame,
                )
            ], 1.0

        if context.input_type == "video":
            video_path = context.raw_data.get("video_path")
            if not video_path:
                raise ValueError("MicroExpressionExtractor requires video_path")
            return self._detect_events_from_video(video_path)

        raise ValueError(
            f"MicroExpressionExtractor supports only video and image profiles, "
            f"got {context.input_type!r}"
        )

    def _read_image(self, image_path: str | Path) -> np.ndarray:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")
        return self._normalize_frame(image)

    def _read_frame_at_index(self, video_path: str | Path, index: int) -> np.ndarray:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(
                    f"Unable to read frame {index} from video: {video_path}"
                )
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return self._normalize_frame(gray)
        finally:
            capture.release()

    def _detect_events_from_video(
        self,
        video_path: str | Path,
    ) -> tuple[list[MicroExpressionEvent], float]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        prev_frame: np.ndarray | None = None
        scores: list[float] = []
        frame_count = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                normalized = self._normalize_frame(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                )
                if prev_frame is not None:
                    scores.append(float(np.mean(np.abs(normalized - prev_frame))))
                prev_frame = normalized
                frame_count += 1
        finally:
            capture.release()

        if frame_count == 0:
            raise ValueError(f"No frames found in video: {video_path}")
        if frame_count == 1:
            single = self._read_frame_at_index(video_path, 0)
            return [
                MicroExpressionEvent(0, 0, 0.0, 0.0, single, single)
            ], fps

        apex_delta_index = int(np.argmax(scores))
        onset_index = max(0, apex_delta_index - 1)
        apex_index = min(frame_count - 1, apex_delta_index + 1)

        # v1 heuristic: one representative onset-apex pair per clip.
        # Replace with AU-based or trained DGM event detection later.
        onset_frame = self._read_frame_at_index(video_path, onset_index)
        apex_frame = self._read_frame_at_index(video_path, apex_index)
        return [
            MicroExpressionEvent(
                onset_index=onset_index,
                apex_index=apex_index,
                start_time=onset_index / fps,
                end_time=apex_index / fps,
                onset_frame=onset_frame,
                apex_frame=apex_frame,
            )
        ], fps

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        resized = cv2.resize(
            frame,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_AREA,
        )
        return resized.astype(np.float32) / 255.0

    def _amplify_weak_signal(self, flow: np.ndarray) -> np.ndarray:
        amplified = flow.copy()
        height, width, _ = amplified.shape

        # Approximate key-AU regions without OpenFace/MediaPipe AU scores:
        # mouth corners/lower face (AU12/AU15) and brow band get higher gain.
        regions = [
            (int(height * 0.55), int(height * 0.85), int(width * 0.20), int(width * 0.80)),
            (int(height * 0.20), int(height * 0.42), int(width * 0.18), int(width * 0.82)),
        ]
        for y0, y1, x0, x1 in regions:
            amplified[y0:y1, x0:x1, :] *= self.amplify_coeff
        return amplified.astype(np.float32)
