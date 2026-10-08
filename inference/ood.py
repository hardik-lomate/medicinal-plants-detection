"""
inference/ood.py
================
Botanical Verification & Diagnostic Status Engine.

Provides transparent diagnostic reporting without overriding the neural network's
authoritative classification prediction on botanical specimens.

Statuses:
- 'identified'                  : Plant identified by model, medicinal monograph in database
- 'identified_no_database_info': Plant identified by model, no monograph in database
- 'unknown'                    : Completely non-plant or out-of-domain image rejected
- 'bad_image'                  : Corrupt or unreadable input image
"""

from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DecisionResult:
    status: str                  # 'identified' | 'identified_no_database_info' | 'unknown' | 'bad_image'
    is_confident: bool
    is_uncertain: bool
    is_ood: bool
    decision_reason: str
    rejection_signals: Dict[str, Any]
    user_message: str


class UncertaintyDecisionEngine:
    """
    Evaluates multi-signal OOD criteria to reject clearly non-plant images
    while preserving authoritative model prediction for genuine botanical specimens.

    OOD Detection Thresholds (empirically calibrated on real plant vs synthetic negatives):

    Calibration data:
      - Plant images (65 samples):  min max_softmax=0.102, mean=0.441
      - OOD images (12 synthetics): max max_softmax=0.049, mean=0.025
      - Full gap: 0.049 (OOD max) to 0.102 (plant min) — no overlap

    Primary threshold = 0.065 (midpoint with 24% safety margin from OOD side)
    Secondary guard: normalized entropy > 0.93 AND confidence < 0.08
    (OOD entropy: 0.945-0.997; plant entropy: 0.06-0.88 — no overlap above 0.93)
    Energy score included in signals for diagnostics (not used for gating to avoid false rejects).
    """

    def __init__(
        self,
        # Primary OOD gate: max softmax below this = uniform distribution = OOD
        ood_softmax_threshold: float = 0.065,
        # Secondary guard: entropy above this (normalized 0-1) = OOD (used with low confidence)
        ood_entropy_threshold: float = 0.930,
        # Legacy parameters kept for backward compat
        min_prediction_confidence: float = 0.10,
        min_margin_threshold: float = 0.02,
        min_stability_threshold: float = 0.30,
        min_embedding_similarity: float = 0.12,
        high_confidence_margin: float = 0.15
    ):
        self.ood_softmax_threshold = ood_softmax_threshold
        self.ood_entropy_threshold = ood_entropy_threshold
        self.min_confidence = min_prediction_confidence
        self.min_margin = min_margin_threshold
        self.min_stability = min_stability_threshold
        self.min_similarity = min_embedding_similarity
        self.high_conf_margin = high_confidence_margin

    def evaluate(
        self,
        confidence: float,
        margin: float,
        stability_score: float,
        quality_score: float,
        quality_acceptable: bool,
        embedding_similarity: float,
        has_database_match: bool,
        has_medicinal_info: bool,
        plant_display_name: str,
        threshold_override: Optional[float] = None,
        # New OOD signal parameters passed from engine.py
        normalized_entropy: Optional[float] = None,
        energy_score: Optional[float] = None,
    ) -> DecisionResult:
        effective_threshold = threshold_override if threshold_override is not None else self.min_confidence

        signals = {
            "confidence": round(confidence, 4),
            "margin": round(margin, 4),
            "stability": round(stability_score, 4),
            "quality_score": round(quality_score, 4),
            "embedding_similarity": round(embedding_similarity, 4),
            "effective_threshold": effective_threshold,
            "normalized_entropy": round(normalized_entropy, 4) if normalized_entropy is not None else None,
            "energy_score": round(energy_score, 3) if energy_score is not None else None,
        }

        # Gate 1: Severely Corrupted / Unusable Image
        if not quality_acceptable and quality_score < 0.15:
            return DecisionResult(
                status="bad_image",
                is_confident=False,
                is_uncertain=True,
                is_ood=False,
                decision_reason="Image failed basic quality validation standards (completely unreadable or blank).",
                rejection_signals=signals,
                user_message="The photograph does not exhibit distinguishable botanical features for reliable identification."
            )

        # Gate 2: True Out-Of-Distribution (Non-Plant Objects)
        #
        # Empirical calibration (65 plant images vs 12 synthetic OOD images):
        #   OOD max softmax:   0.049 (logo/stripe — highest observed)
        #   Plant min softmax: 0.102 (Mango — lowest observed among 65 plant samples)
        #   Full gap — no overlap exists.
        #
        # Primary signal: max softmax < 0.065
        #   - Detects near-uniform softmax distributions (OOD characteristic)
        #   - 24% safety margin above max OOD; 36% safety margin below min plant
        #
        # Secondary signal: normalized entropy > 0.93 AND confidence < 0.08
        #   - OOD entropy: 0.945-0.997; plants: 0.06-0.88 — complete gap above 0.93
        #   - Only fires if primary gate was narrowly missed (confidence 0.065-0.08)

        is_ood = False
        ood_reason = ""

        if confidence < self.ood_softmax_threshold:
            is_ood = True
            inv_conf = 1.0 / max(confidence, 0.001)
            ood_reason = (
                f"Near-uniform probability distribution (max softmax: {confidence:.3f} < "
                f"threshold {self.ood_softmax_threshold:.3f}, equivalent to >{inv_conf:.0f} equally "
                f"likely classes). Input does not match any cataloged botanical species."
            )
        elif (
            normalized_entropy is not None and
            normalized_entropy > self.ood_entropy_threshold and
            confidence < 0.08
        ):
            is_ood = True
            ood_reason = (
                f"Near-maximum classification entropy ({normalized_entropy:.3f} > "
                f"{self.ood_entropy_threshold:.3f}) with low confidence ({confidence:.3f}). "
                "Input does not resemble any cataloged botanical morphology."
            )

        if is_ood:
            return DecisionResult(
                status="unknown",
                is_confident=False,
                is_uncertain=True,
                is_ood=True,
                decision_reason=ood_reason,
                rejection_signals=signals,
                user_message=(
                    "The uploaded image does not appear to contain a recognizable plant or leaf. "
                    "Please upload a clear photograph of a plant or leaf specimen."
                )
            )

        # Gate 3: Plant Identification (Model argmax is authoritative for botanical images)
        if has_database_match and has_medicinal_info:
            final_status = "identified"
            msg = f"Successfully identified as {plant_display_name} with verified pharmacological monograph."
        else:
            final_status = "identified_no_database_info"
            msg = (
                f"Plant identified as {plant_display_name}. "
                "No verified medicinal information is available for this plant in our database."
            )

        # Informative diagnostic note for close calls
        is_low_confidence = (confidence < effective_threshold) or (margin < self.min_margin)
        decision_reason = (
            f"Close competition or low confidence ({confidence:.1%}, margin: +{margin:.1%}), displaying top-1 match."
            if is_low_confidence
            else "Authoritative model prediction passed certainty verification."
        )

        return DecisionResult(
            status=final_status,
            is_confident=True,
            is_uncertain=is_low_confidence,
            is_ood=False,
            decision_reason=decision_reason,
            rejection_signals=signals,
            user_message=msg
        )


# Default shared decision engine instance
DEFAULT_DECISION_ENGINE = UncertaintyDecisionEngine()
