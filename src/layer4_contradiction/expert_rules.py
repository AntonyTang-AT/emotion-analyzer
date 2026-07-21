"""Expert-rule contradiction type interpreter for L4.3.

Classifies masking / sarcasm / hidden_emotion / intensity_mismatch / consistent
from modality valence values. Output is an explanation and weight prior hint
only; it must not bypass disagreement_score or routing confidence.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.core.types import MODALITIES, ContradictionType, VAConfidence
from src.utils.config_loader import load_config

DEFAULT_MASKING_TEXT_V_MIN = 0.6
DEFAULT_MASKING_MICRO_V_MAX = -0.4
DEFAULT_SARCASM_TEXT_V_MIN = 0.3
DEFAULT_SARCASM_SPEECH_V_MAX = -0.4
DEFAULT_SARCASM_MACRO_V_MAX = -0.4
DEFAULT_HIDDEN_MACRO_V_ABS_MAX = 0.2
DEFAULT_HIDDEN_MICRO_V_ABS_MIN = 0.5
DEFAULT_INTENSITY_VA_DIFF_MIN = 0.6


@dataclass(frozen=True)
class ExpertRulesConfig:
    """Thresholds for L4.3 expert rules (from ``models.yaml`` ``layer4.expert_rules``)."""

    masking_text_v_min: float = DEFAULT_MASKING_TEXT_V_MIN
    masking_micro_v_max: float = DEFAULT_MASKING_MICRO_V_MAX
    sarcasm_text_v_min: float = DEFAULT_SARCASM_TEXT_V_MIN
    sarcasm_speech_v_max: float = DEFAULT_SARCASM_SPEECH_V_MAX
    sarcasm_macro_v_max: float = DEFAULT_SARCASM_MACRO_V_MAX
    hidden_macro_v_abs_max: float = DEFAULT_HIDDEN_MACRO_V_ABS_MAX
    hidden_micro_v_abs_min: float = DEFAULT_HIDDEN_MICRO_V_ABS_MIN
    intensity_va_diff_min: float = DEFAULT_INTENSITY_VA_DIFF_MIN

    @classmethod
    def from_models(
        cls,
        models_config: Mapping[str, Any] | None = None,
    ) -> ExpertRulesConfig:
        if models_config is None:
            models_config = load_config("models")
        rules = models_config.get("layer4", {}).get("expert_rules", {})
        masking = rules.get("masking", {})
        sarcasm = rules.get("sarcasm", {})
        hidden = rules.get("hidden_emotion", {})
        intensity = rules.get("intensity_mismatch", {})
        return cls(
            masking_text_v_min=_finite_float(
                masking.get("text_v_min", DEFAULT_MASKING_TEXT_V_MIN),
                "masking.text_v_min",
            ),
            masking_micro_v_max=_finite_float(
                masking.get("micro_v_max", DEFAULT_MASKING_MICRO_V_MAX),
                "masking.micro_v_max",
            ),
            sarcasm_text_v_min=_finite_float(
                sarcasm.get("text_v_min", DEFAULT_SARCASM_TEXT_V_MIN),
                "sarcasm.text_v_min",
            ),
            sarcasm_speech_v_max=_finite_float(
                sarcasm.get("speech_v_max", DEFAULT_SARCASM_SPEECH_V_MAX),
                "sarcasm.speech_v_max",
            ),
            sarcasm_macro_v_max=_finite_float(
                sarcasm.get("macro_v_max", DEFAULT_SARCASM_MACRO_V_MAX),
                "sarcasm.macro_v_max",
            ),
            hidden_macro_v_abs_max=_finite_float(
                hidden.get("macro_v_abs_max", DEFAULT_HIDDEN_MACRO_V_ABS_MAX),
                "hidden_emotion.macro_v_abs_max",
            ),
            hidden_micro_v_abs_min=_finite_float(
                hidden.get("micro_v_abs_min", DEFAULT_HIDDEN_MICRO_V_ABS_MIN),
                "hidden_emotion.micro_v_abs_min",
            ),
            intensity_va_diff_min=_finite_float(
                intensity.get("va_diff_min", DEFAULT_INTENSITY_VA_DIFF_MIN),
                "intensity_mismatch.va_diff_min",
            ),
        )


@dataclass(frozen=True)
class ExpertRuleResult:
    """Contradiction type and the modalities that triggered the matched rule."""

    contradiction_type: ContradictionType
    involved_modalities: list[str]
    matched_rule: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "contradiction_type": self.contradiction_type.value,
            "involved_modalities": list(self.involved_modalities),
            "matched_rule": self.matched_rule,
        }


def same_sign(a: float, b: float) -> bool:
    """Return True when ``a`` and ``b`` share a non-negative or non-positive sign.

    Zero is treated as matching either side (``a * b >= 0``).
    """
    return float(a) * float(b) >= 0.0


def classify_contradiction(
    modality_va: Mapping[str, Any],
    *,
    config: ExpertRulesConfig | None = None,
) -> ExpertRuleResult:
    """Classify contradiction type from modality valence values.

    Rules are evaluated in fixed priority order. A rule whose required
    modalities are missing is skipped (values are never silently filled with
    0). Sarcasm treats speech/macro as an OR: at least one present negative
    modality is enough. When no rule matches, returns ``consistent`` with an
    empty ``involved_modalities`` list.
    """
    rules = config if config is not None else ExpertRulesConfig.from_models()
    valences = _extract_valences(modality_va)

    text_v = valences.get("text")
    speech_v = valences.get("speech")
    macro_v = valences.get("macro")
    micro_v = valences.get("micro")

    if text_v is not None and micro_v is not None:
        if text_v > rules.masking_text_v_min and micro_v < rules.masking_micro_v_max:
            return _result(ContradictionType.MASKING, ["text", "micro"])

    if text_v is not None and text_v > rules.sarcasm_text_v_min:
        involved = ["text"]
        if speech_v is not None and speech_v < rules.sarcasm_speech_v_max:
            involved.append("speech")
        if macro_v is not None and macro_v < rules.sarcasm_macro_v_max:
            involved.append("macro")
        if len(involved) > 1:
            return _result(ContradictionType.SARCASM, involved)

    if macro_v is not None and micro_v is not None:
        if (
            abs(macro_v) < rules.hidden_macro_v_abs_max
            and abs(micro_v) > rules.hidden_micro_v_abs_min
        ):
            return _result(ContradictionType.HIDDEN_EMOTION, ["macro", "micro"])

    if text_v is not None and micro_v is not None:
        if (
            same_sign(text_v, micro_v)
            and abs(text_v - micro_v) > rules.intensity_va_diff_min
        ):
            return _result(ContradictionType.INTENSITY_MISMATCH, ["text", "micro"])

    return ExpertRuleResult(
        contradiction_type=ContradictionType.CONSISTENT,
        involved_modalities=[],
        matched_rule=None,
    )


def _result(
    contradiction_type: ContradictionType,
    involved: Sequence[str],
) -> ExpertRuleResult:
    ordered = [modality for modality in MODALITIES if modality in involved]
    return ExpertRuleResult(
        contradiction_type=contradiction_type,
        involved_modalities=ordered,
        matched_rule=contradiction_type.value,
    )


def _extract_valences(modality_va: Mapping[str, Any]) -> dict[str, float]:
    unknown = [name for name in modality_va if name not in MODALITIES]
    if unknown:
        raise ValueError(f"unknown modalities: {unknown}")

    valences: dict[str, float] = {}
    for modality in MODALITIES:
        if modality not in modality_va:
            continue
        valences[modality] = _coerce_valence(modality_va[modality], modality)
    return valences


def _coerce_valence(value: Any, modality: str) -> float:
    if isinstance(value, VAConfidence):
        valence = value.valence
    elif isinstance(value, Mapping):
        try:
            valence = value["valence"]
        except KeyError as exc:
            raise ValueError(
                f"VA value for {modality!r} must contain valence"
            ) from exc
    else:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError(
                f"VA value for {modality!r} must be VAConfidence, "
                "a mapping with valence, or a (v, a) pair"
            )
        if len(value) < 1:
            raise ValueError(f"VA value for {modality!r} must contain valence")
        valence = value[0]

    return _finite_valence(valence, modality)


def _finite_valence(value: Any, modality: str) -> float:
    try:
        valence = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"valence for {modality!r} must be numeric") from exc
    if not math.isfinite(valence) or not -1.0 <= valence <= 1.0:
        raise ValueError(
            f"valence for {modality!r} must be a finite value in [-1, 1]"
        )
    return valence


def _finite_float(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


__all__ = [
    "ExpertRuleResult",
    "ExpertRulesConfig",
    "classify_contradiction",
    "same_sign",
]
