"""L1 speech feature extraction: Wav2Vec2 + FoX pooling + prosody."""

from __future__ import annotations

from typing import Any

import librosa
import numpy as np

from src.core.context import DataContext
from src.core.interfaces import FeatureExtractor
from src.core.types import FeatureDict
from src.utils.audio_utils import (
    extract_audio_from_video,
    frame_audio,
    load_audio,
)
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

PROSODY_DIM = 16  # f0(1) + energy(1) + zcr(1) + mfcc(13)

# Shared Wav2Vec2 weights keyed by model name (avoid reload per pipeline run).
_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}


class SpeechExtractor(FeatureExtractor):
    """Extract per-second speech vectors from audio or video input."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        speech_cfg = (config or load_config("features"))["speech"]
        self._model_name: str = speech_cfg["wav2vec2_model"]
        self._sample_rate: int = int(speech_cfg["sample_rate"])
        self._embedding_dim: int = int(speech_cfg["embedding_dim"])
        fox_cfg = speech_cfg.get("fox", {})
        self._fox_enabled: bool = bool(fox_cfg.get("enabled", True))
        self._fox_gamma: float = float(fox_cfg.get("gamma", 0.1))
        prosody_cfg = speech_cfg.get("prosody", {})
        self._prosody_enabled: bool = bool(prosody_cfg.get("enabled", True))
        self._feature_dim = self._embedding_dim + (
            PROSODY_DIM if self._prosody_enabled else 0
        )

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def _get_model(self) -> tuple[Any, Any]:
        cached = _MODEL_CACHE.get(self._model_name)
        if cached is not None:
            return cached

        import torch
        from transformers import Wav2Vec2Model, Wav2Vec2Processor

        processor = Wav2Vec2Processor.from_pretrained(self._model_name)
        model = Wav2Vec2Model.from_pretrained(self._model_name)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        _MODEL_CACHE[self._model_name] = (model, processor)
        return model, processor

    def _resolve_waveform(self, context: DataContext) -> tuple[np.ndarray, int]:
        audio_path = context.raw_data.get("audio_path")
        if audio_path:
            return load_audio(audio_path, target_sr=self._sample_rate)

        video_path = context.raw_data.get("video_path")
        if video_path:
            waveform = extract_audio_from_video(
                video_path, target_sr=self._sample_rate
            )
            if isinstance(waveform, np.ndarray):
                return waveform.astype(np.float32), self._sample_rate
            return load_audio(waveform, target_sr=self._sample_rate)

        raise ValueError(
            "SpeechExtractor requires audio_path or video_path in context.raw_data"
        )

    def _wav2vec_frames(self, chunk: np.ndarray, sample_rate: int) -> np.ndarray:
        import torch

        model, processor = self._get_model()
        inputs = processor(
            chunk,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            hidden = model(**inputs).last_hidden_state.squeeze(0)
        return hidden.cpu().numpy().astype(np.float32)

    def fox_pool(self, frame_feats: np.ndarray) -> np.ndarray:
        """Apply simplified FoX: exponential decay weighting over time steps."""
        if frame_feats.ndim != 2:
            raise ValueError("frame_feats must be 2-D (time, dim)")

        if frame_feats.shape[0] == 0:
            return np.zeros(frame_feats.shape[1], dtype=np.float32)

        if not self._fox_enabled:
            return frame_feats.mean(axis=0).astype(np.float32)

        num_frames = frame_feats.shape[0]
        indices = np.arange(num_frames, dtype=np.float32)
        delta_t = (num_frames - 1) - indices
        weights = np.exp(-self._fox_gamma * delta_t)
        weights /= weights.sum()
        pooled = (frame_feats * weights[:, np.newaxis]).sum(axis=0)
        return pooled.astype(np.float32)

    def prosody_vector(self, chunk: np.ndarray, sample_rate: int) -> np.ndarray:
        """Return 16-dim prosody vector: f0, energy, zcr, mfcc(13)."""
        if len(chunk) == 0:
            return np.zeros(PROSODY_DIM, dtype=np.float32)

        f0_value = 0.0
        try:
            f0, _, _ = librosa.pyin(
                chunk,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=sample_rate,
            )
            voiced = f0[~np.isnan(f0)] if f0 is not None else np.array([])
            if voiced.size > 0:
                f0_value = float(np.mean(voiced))
        except Exception:
            logger.debug("F0 extraction failed; using 0.0", exc_info=True)

        energy = float(np.mean(librosa.feature.rms(y=chunk)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(chunk)))
        mfcc = librosa.feature.mfcc(y=chunk, sr=sample_rate, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)

        return np.concatenate(
            [
                np.array([f0_value, energy, zcr], dtype=np.float32),
                mfcc_mean.astype(np.float32),
            ]
        )

    def extract(self, context: DataContext) -> FeatureDict:
        waveform, sample_rate = self._resolve_waveform(context)
        if len(waveform) == 0:
            logger.warning("Empty waveform; returning no speech features")
            return {"speech": []}

        frames = frame_audio(
            waveform,
            sample_rate,
            frame_length_sec=1.0,
            hop_length_sec=1.0,
        )
        results: list[dict[str, Any]] = []
        for chunk, timestamp in frames:
            w2v_frames = self._wav2vec_frames(chunk, sample_rate)
            embedding = self.fox_pool(w2v_frames)
            if embedding.shape[0] != self._embedding_dim:
                raise ValueError(
                    f"Expected Wav2Vec2 dim {self._embedding_dim}, "
                    f"got {embedding.shape[0]}"
                )

            if self._prosody_enabled:
                prosody = self.prosody_vector(chunk, sample_rate)
                speech_feature = np.concatenate([embedding, prosody]).astype(
                    np.float32
                )
            else:
                speech_feature = embedding.astype(np.float32)

            results.append(
                {
                    "speech_feature": speech_feature,
                    "timestamp": float(timestamp),
                }
            )

        return {"speech": results}
