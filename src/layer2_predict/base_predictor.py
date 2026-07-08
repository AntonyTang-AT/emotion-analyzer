"""L2 VA predictor abstract base class."""

from __future__ import annotations

from abc import ABC

from src.core.interfaces import VAPredictor


class BasePredictor(VAPredictor, ABC):
    """Base class for single-modality two-branch VA predictors.

    Subclasses implement ``predict_self`` and ``predict_inter`` accepting a
    1-D feature vector (``np.ndarray`` shape ``(input_dim,)`` or ``list[float]``)
    and returning a ``VAConfidence`` instance.
    """

    modality: str
