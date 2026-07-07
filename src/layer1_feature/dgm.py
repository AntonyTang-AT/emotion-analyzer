"""Displacement-field generation for micro-expression onset/apex pairs."""

from __future__ import annotations

import cv2
import numpy as np


class DisplacementFieldGenerator:
    """Compute a dense 2-D displacement field between two aligned frames."""

    def __init__(self, *, size: int = 112) -> None:
        self.size = int(size)

    def compute(self, onset: np.ndarray, apex: np.ndarray) -> np.ndarray:
        onset_gray = self._prepare(onset)
        apex_gray = self._prepare(apex)

        flow = self._tvl1(onset_gray, apex_gray)
        if flow is None:
            flow = cv2.calcOpticalFlowFarneback(
                onset_gray,
                apex_gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
        return flow.astype(np.float32)

    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(frame, (self.size, self.size), interpolation=cv2.INTER_AREA)
        if resized.dtype != np.float32:
            resized = resized.astype(np.float32)
        if resized.max(initial=0.0) > 1.0:
            resized = resized / 255.0
        return resized

    def _tvl1(self, onset: np.ndarray, apex: np.ndarray) -> np.ndarray | None:
        factory = None
        if hasattr(cv2, "optflow") and hasattr(cv2.optflow, "DualTVL1OpticalFlow_create"):
            factory = cv2.optflow.DualTVL1OpticalFlow_create
        elif hasattr(cv2, "DualTVL1OpticalFlow_create"):
            factory = cv2.DualTVL1OpticalFlow_create

        if factory is None:
            return None

        tvl1 = factory()
        return tvl1.calc(onset, apex, None)
