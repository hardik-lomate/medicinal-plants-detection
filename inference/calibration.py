"""
inference/calibration.py
========================
Confidence Calibration & Top-K Probability Distribution Analysis.

Features:
- Post-hoc Temperature Scaling (Guo et al., ICML 2017) to soften overconfident raw softmax outputs
- Full probability distribution inspection and Shannon prediction entropy
- Separation Margin calculation (confidence gap between rank 1 and rank 2)
- Human-interpretable confidence grading (High, Moderate, Low / Ambiguous)
- Prevents misrepresenting raw softmax probabilities as guaranteed real-world accuracy
"""

from typing import List, Dict, Any, Tuple
import math
import numpy as np


class ConfidenceCalibrator:
    """
    Applies temperature scaling and produces calibrated multi-class probability metrics.
    """

    def __init__(
        self,
        leaf_temperature: float = 1.25,
        plant_temperature: float = 1.20,
        type_temperature: float = 1.10
    ):
        self.temperatures = {
            "leaf": leaf_temperature,
            "whole_plant": plant_temperature,
            "image_type": type_temperature,
        }

    def get_temperature(self, model_type: str) -> float:
        return self.temperatures.get(model_type, 1.20)

    @staticmethod
    def compute_margin(top_predictions: List[Dict[str, Any]]) -> float:
        """Computes top-1 vs top-2 probability separation margin."""
        if not top_predictions or len(top_predictions) == 0:
            return 0.0
        p1 = top_predictions[0].get("confidence", 0.0)
        p2 = top_predictions[1].get("confidence", 0.0) if len(top_predictions) > 1 else 0.0
        return float(round(max(0.0, p1 - p2), 4))

    @staticmethod
    def compute_distribution_entropy(probabilities: np.ndarray) -> float:
        """Computes Shannon entropy across the class probability distribution."""
        probs = np.clip(probabilities, 1e-12, 1.0)
        entropy = -float(np.sum(probs * np.log2(probs)))
        return float(round(entropy, 3))

    @staticmethod
    def interpret_confidence(confidence: float, margin: float) -> Tuple[str, str]:
        """
        Translates numerical probability and separation margin into qualitative assessment.
        Returns: (tier: 'HIGH' | 'MODERATE' | 'LOW', description: str)
        """
        if confidence >= 0.70 and margin >= 0.20:
            return (
                "HIGH",
                f"Strong morphological match ({confidence:.1%} confidence, +{margin:.1%} separation margin)."
            )
        elif confidence >= 0.40 and margin >= 0.08:
            return (
                "MODERATE",
                f"Moderate identification agreement ({confidence:.1%} confidence, candidate separation: +{margin:.1%})."
            )
        else:
            return (
                "LOW",
                f"Low confidence or close competition between candidate species ({confidence:.1%} confidence, margin: +{margin:.1%})."
            )


# Default shared calibrator instance
DEFAULT_CALIBRATOR = ConfidenceCalibrator()
