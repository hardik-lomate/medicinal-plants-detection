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
    Evaluates multi-signal criteria to produce robust identification decisions.
    Preserves the model prediction as authoritative on botanical images.
    """

    def __init__(
        self,
        min_prediction_confidence: float = 0.10,
        min_margin_threshold: float = 0.02,
        min_stability_threshold: float = 0.30,
        min_embedding_similarity: float = 0.12,
        high_confidence_margin: float = 0.15
    ):
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
        threshold_override: Optional[float] = None
    ) -> DecisionResult:
        effective_threshold = threshold_override if threshold_override is not None else self.min_confidence

        signals = {
            "confidence": round(confidence, 4),
            "margin": round(margin, 4),
            "stability": round(stability_score, 4),
            "quality_score": round(quality_score, 4),
            "embedding_similarity": round(embedding_similarity, 4),
            "effective_threshold": effective_threshold
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
        # Only triggers when features show near-zero botanical alignment across all indicators
        is_ood = (
            confidence < 0.08 and
            margin < 0.02 and
            embedding_similarity < 0.10 and
            quality_score < 0.35
        )

        if is_ood:
            return DecisionResult(
                status="unknown",
                is_confident=False,
                is_uncertain=True,
                is_ood=True,
                decision_reason="Specimen features show zero alignment with cataloged botanical species manifold.",
                rejection_signals=signals,
                user_message="The photograph does not sufficiently match any cataloged botanical species."
            )

        # Gate 3: Plant Identification (Model argmax is authoritative)
        if has_database_match and has_medicinal_info:
            final_status = "identified"
            msg = f"Successfully identified as {plant_display_name} with verified pharmacological monograph."
        else:
            final_status = "identified_no_database_info"
            msg = f"Plant identified as {plant_display_name}. No verified medicinal information is available for this plant in our database."

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
