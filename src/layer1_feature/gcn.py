"""Lightweight multi-view projection for micro-expression displacement fields."""

from __future__ import annotations

import cv2
import numpy as np


class MultiViewMicroProjector:
    """Fuse content and relation views into a 256-dim micro feature."""

    def __init__(self, *, content_dim: int = 128, relation_dim: int = 128) -> None:
        self.content_dim = int(content_dim)
        self.relation_dim = int(relation_dim)

    def project(self, flow: np.ndarray) -> np.ndarray:
        if flow.ndim != 3 or flow.shape[2] != 2:
            raise ValueError("flow must have shape (height, width, 2)")

        content = self._content_view(flow)
        relation = self._relation_view(flow)
        return np.concatenate([content, relation]).astype(np.float32)

    def _content_view(self, flow: np.ndarray) -> np.ndarray:
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
        stats = np.array(
            [
                magnitude.mean(),
                magnitude.std(),
                magnitude.max(initial=0.0),
                np.percentile(magnitude, 25),
                np.percentile(magnitude, 50),
                np.percentile(magnitude, 75),
                np.sin(angle).mean(),
                np.cos(angle).mean(),
            ],
            dtype=np.float32,
        )

        hist, _ = np.histogram(magnitude, bins=32, range=(0.0, max(float(magnitude.max()), 1e-6)))
        hist = hist.astype(np.float32)
        if hist.sum() > 0:
            hist /= hist.sum()

        pooled = cv2.resize(
            magnitude,
            (8, 11),
            interpolation=cv2.INTER_AREA,
        ).reshape(-1).astype(np.float32)

        vector = np.concatenate([stats, hist, pooled])
        return self._fit_dim(vector, self.content_dim)

    def _relation_view(self, flow: np.ndarray) -> np.ndarray:
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
        grid = cv2.resize(magnitude, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32)

        # Two rounds of grid-neighborhood smoothing approximate a tiny GCN over
        # facial regions without needing landmark dependencies.
        smoothed = grid.copy()
        for _ in range(2):
            padded = np.pad(smoothed, 1, mode="edge")
            smoothed = (
                padded[1:-1, 1:-1]
                + padded[:-2, 1:-1]
                + padded[2:, 1:-1]
                + padded[1:-1, :-2]
                + padded[1:-1, 2:]
            ) / 5.0

        horizontal = np.diff(smoothed, axis=1, prepend=smoothed[:, :1])
        vertical = np.diff(smoothed, axis=0, prepend=smoothed[:1, :])
        vector = np.concatenate(
            [
                smoothed.reshape(-1),
                horizontal.reshape(-1),
                vertical.reshape(-1),
            ]
        )
        return self._fit_dim(vector, self.relation_dim)

    def _fit_dim(self, vector: np.ndarray, dim: int) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vector.size >= dim:
            fitted = vector[:dim]
        else:
            fitted = np.pad(vector, (0, dim - vector.size), mode="constant")

        norm = float(np.linalg.norm(fitted))
        if norm > 0.0:
            fitted = fitted / norm
        return fitted.astype(np.float32)
