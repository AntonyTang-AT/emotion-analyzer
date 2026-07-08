"""L1 text feature extraction: Whisper ASR + BERT sentence embeddings."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from src.core.context import DataContext
from src.core.interfaces import FeatureExtractor
from src.core.types import FeatureDict, InputType, TextSubtype
from src.utils.audio_utils import extract_audio_from_video
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

_WHISPER_CACHE: dict[str, Any] = {}
_BERT_CACHE: dict[str, tuple[Any, Any]] = {}


class TextExtractor(FeatureExtractor):
    """Extract timestamped text embeddings from audio, video, or raw text."""

    modality = "text"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        features_cfg = config or load_config("features")
        text_cfg = features_cfg["text"]
        speech_cfg = features_cfg.get("speech", {})

        self._whisper_model_name: str = str(text_cfg["whisper_model"])
        self._bert_model_name: str = str(text_cfg["bert_model"])
        self._embedding_dim: int = int(text_cfg.get("embedding_dim", 768))
        self._language: str | None = text_cfg.get("language", "auto")
        if self._language == "auto":
            self._language = None
        self._sample_rate: int = int(speech_cfg.get("sample_rate", 16000))

    @property
    def feature_dim(self) -> int:
        return self._embedding_dim

    def extract(self, context: DataContext) -> FeatureDict:
        if context.input_type == InputType.TEXT.value:
            segments = self._segments_from_text_context(context)
        elif context.input_type in {InputType.AUDIO.value, InputType.VIDEO.value}:
            segments = self._segments_from_media_context(context)
        else:
            logger.info("TextExtractor skipped for input_type=%s", context.input_type)
            return {"text": []}

        results: list[dict[str, Any]] = []
        for segment in segments:
            text = str(segment["text"]).strip()
            if not text:
                continue
            embedding = self._embed_text(text)
            if embedding.shape != (self._embedding_dim,):
                raise ValueError(
                    f"Expected BERT embedding shape ({self._embedding_dim},), "
                    f"got {embedding.shape}"
                )
            results.append(
                {
                    "text_embedding": embedding.astype(np.float32, copy=False),
                    "start_time": float(segment["start_time"]),
                    "end_time": float(segment["end_time"]),
                    "text": text,
                }
            )

        return {"text": results}

    def _segments_from_media_context(
        self, context: DataContext
    ) -> list[dict[str, float | str]]:
        if context.input_type == InputType.VIDEO.value:
            video_path = context.raw_data.get("video_path")
            if not video_path:
                raise ValueError("TextExtractor requires video_path for video input")
            audio_source = extract_audio_from_video(
                video_path,
                target_sr=self._sample_rate,
            )
        else:
            audio_source = context.raw_data.get("audio_path")
            if not audio_source:
                raise ValueError("TextExtractor requires audio_path for audio input")

        transcription = self._transcribe_audio(audio_source)
        return self._segments_from_transcription(transcription)

    def _segments_from_text_context(
        self, context: DataContext
    ) -> list[dict[str, float | str]]:
        text = context.raw_data.get("text_content")
        if text is None:
            text_path = context.raw_data.get("text_path")
            if not text_path:
                raise ValueError(
                    "TextExtractor requires text_content or text_path for text input"
                )
            text = Path(text_path).read_text(encoding="utf-8")

        parts = self._split_text(str(text), context.text_subtype)
        return self._assign_logical_times(parts)

    def _split_text(self, text: str, text_subtype: str | None) -> list[str]:
        text = text.strip()
        if not text:
            return []

        if text_subtype == TextSubtype.DIALOGUE.value:
            lines = [self._strip_speaker_prefix(line) for line in text.splitlines()]
            turns = [line.strip() for line in lines if line.strip()]
            if len(turns) > 1:
                return turns
            return self._split_sentences(text)

        paragraphs = re.split(r"(?:\r?\n\s*){2,}", text)
        parts = [part.strip() for part in paragraphs if part.strip()]
        return parts or [text]

    @staticmethod
    def _strip_speaker_prefix(line: str) -> str:
        return re.sub(r"^\s*[^:：]{1,24}[:：]\s*", "", line).strip()

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        parts = re.split(r"(?<=[。！？!?\.])\s+", text.strip())
        if len(parts) == 1:
            parts = re.split(r"(?<=[。！？!?\.])", text.strip())
        return [part.strip() for part in parts if part.strip()]

    @staticmethod
    def _assign_logical_times(parts: list[str]) -> list[dict[str, float | str]]:
        if not parts:
            return []
        if len(parts) == 1:
            return [{"text": parts[0], "start_time": 0.0, "end_time": 1.0}]
        return [
            {
                "text": part,
                "start_time": float(index),
                "end_time": float(index + 1),
            }
            for index, part in enumerate(parts)
        ]

    @staticmethod
    def _segments_from_transcription(
        transcription: dict[str, Any]
    ) -> list[dict[str, float | str]]:
        raw_segments = transcription.get("segments")
        if isinstance(raw_segments, list) and raw_segments:
            segments: list[dict[str, float | str]] = []
            for item in raw_segments:
                text = str(item.get("text", "")).strip()
                if not text:
                    continue
                start = float(item.get("start", 0.0))
                end = float(item.get("end", start))
                segments.append({"text": text, "start_time": start, "end_time": end})
            if segments:
                return segments

        text = str(transcription.get("text", "")).strip()
        return [{"text": text, "start_time": 0.0, "end_time": 1.0}] if text else []

    def _transcribe_audio(self, audio_source: str | Path | np.ndarray) -> dict[str, Any]:
        model = self._get_whisper_model()
        kwargs: dict[str, Any] = {}
        if self._language is not None:
            kwargs["language"] = self._language
        return model.transcribe(audio_source, **kwargs)

    def _get_whisper_model(self) -> Any:
        cached = _WHISPER_CACHE.get(self._whisper_model_name)
        if cached is not None:
            return cached

        import whisper

        model = whisper.load_model(self._whisper_model_name)
        _WHISPER_CACHE[self._whisper_model_name] = model
        return model

    def _embed_text(self, text: str) -> np.ndarray:
        import torch

        model, tokenizer = self._get_bert_model()
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = model(**inputs)
        embedding = output.last_hidden_state[:, 0, :].squeeze(0)
        return embedding.detach().cpu().numpy().astype(np.float32)

    def _get_bert_model(self) -> tuple[Any, Any]:
        cached = _BERT_CACHE.get(self._bert_model_name)
        if cached is not None:
            return cached

        import torch
        from transformers import BertModel, BertTokenizer

        tokenizer = BertTokenizer.from_pretrained(self._bert_model_name)
        model = BertModel.from_pretrained(self._bert_model_name)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        _BERT_CACHE[self._bert_model_name] = (model, tokenizer)
        return model, tokenizer
