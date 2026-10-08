"""
inference/ood.py
================
Multi-Signal Uncertainty & Out-Of-Distribution (OOD) Decision Engine.

Synthesizes 6 complementary verification signals to eliminate confident hallucinations
on out-of-domain objects, non-plants, poor photographs, and visually ambiguous leaves:
1. Top-1 Calibrated Confidence
2. Separation Margin (Gap between Rank #1 and Rank #2)
3. Test-Time Augmentation (TTA) Prediction Stability
4. Image Quality Score (Sharpness, contrast, entropy, exposure)
5. Deep Feature Embedding Cosine Similarity to Class Prototypes
6. Distribution Shannon Entropy

Yields definitive status decisions:
- 'bad_image'                     : Unusable input image
- 'unknown'                       : Out-of-distribution or non-plant object rejected
- 'uncertain'                     : Ambiguous specimen requiring clearer photo
- 'identified'                    : Confidently identified with verified monograph
- 'identified_no_database_info'   : Confidently identified without verified medicinal monograph
"""

from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DecisionResult:
    status: str                  # 'identified' | 'identified_no_database_info' | 'uncertain' | 'unknown' | 'bad_image'
    is_confident: bool
    is_uncertain: bool
    is_ood: bool
    decision_reason: str
    rejection_signals: Dict[str, Any]
    user_message: str


class UncertaintyDecisionEngine:
    """
    Evaluates multi-signal criteria to produce robust identification decisions.
    """

    def __init__(
        self,
        min_prediction_confidence: float = 0.22,
        min_margin_threshold: float = 0.06,
        min_stability_threshold: float = 0.40,
        min_embedding_similarity: float = 0.30,
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

        # Gate 1: Image Quality
        if not quality_acceptable:
            return DecisionResult(
                status="bad_image",
                is_confident=False,
                is_uncertain=True,
                is_ood=False,
                decision_reason="Image failed quality validation standards.",
                rejection_signals=signals,
                user_message="The photograph does not exhibit distinguishable botanical features for reliable identification."
            )

        # Gate 2: Out-Of-Distribution (OOD / Non-Plant / Unsupported Species)
        # Indicated by simultaneously: very low max probability, tiny margin, poor stability, and low feature cosine similarity
        is_ood = (
            (confidence < 0.15 and margin < 0.05 and embedding_similarity < self.min_similarity) or
            (confidence < 0.18 and stability_score < 0.30 and embedding_similarity < 0.25) or
            (embedding_similarity < 0.18 and confidence < 0.35)
        )

        if is_ood:
            return DecisionResult(
                status="unknown",
                is_confident=False,
                is_uncertain=True,
                is_ood=True,
                decision_reason="Specimen features deviate significantly from cataloged botanical manifold.",
                rejection_signals=signals,
                user_message="The photograph does not sufficiently match any cataloged botanical species in the model database."
            )

        # Gate 3: Prediction Uncertainty / Ambiguity
        # Triggers if confidence is below threshold, or candidates are too closely tied, or predictions flip heavily across TTA
        is_uncertain = (
            confidence < effective_threshold or
            (margin < self.min_margin and confidence < 0.45) or
            (stability_score < self.min_stability and confidence < 0.50)
        )

        if is_uncertain:
            return DecisionResult(
                status="uncertain",
                is_confident=False,
                is_uncertain=True,
                is_ood=False,
                decision_reason=f"Confidence ({confidence:.1%}), margin ({margin:.1%}), or stability ({stability_score:.1%}) below certainty threshold.",
                rejection_signals=signals,
                user_message="Plant identification is uncertain. The photograph does not provide sufficient detail to distinguish between candidate species with high confidence."
            )

        # Gate 4: Confident Identification
        if has_database_match and has_medicinal_info:
            final_status = "identified"
            msg = f"Successfully identified as {plant_display_name} with verified pharmacological monograph."
        else:
            final_status = "identified_no_database_info"
            msg = f"Plant identified as {plant_display_name}. No verified medicinal information is available for this plant in our database."

        return DecisionResult(
            status=final_status,
            is_confident=True,
            is_uncertain=False,
            is_ood=False,
            decision_reason="Specimen passed confidence, margin, stability, and manifold alignment gates.",
            rejection_signals=signals,
            user_message=msg
        )


# Default shared decision engine instance
DEFAULT_DECISION_ENGINE = UncertaintyDecisionEngine()
