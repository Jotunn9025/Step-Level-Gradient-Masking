"""Default threshold configurations for masking signals.

These thresholds were calibrated by running the base Qwen3-4B model and
the baseline GRPO model on the MATH-500 evaluation set, identifying
divergent items (correct→incorrect or vice versa), and computing the
average per-step signal on those items.

Calibration scripts: scripts/calibrate_threshold.py
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ThresholdConfig:
    """Configuration for a masking threshold.

    Attributes:
        signal_type: Which masking signal this threshold applies to.
        value: The threshold value.
        calibration_dataset: Dataset used for calibration.
        calibration_method: How the threshold was derived.
    """
    signal_type: str
    value: float
    calibration_dataset: str = "MATH-500"
    calibration_method: str = "mean_on_divergent_items"


# ── Default thresholds (calibrated on Qwen3-4B) ──

KL_THRESHOLD = ThresholdConfig(
    signal_type="kl_divergence",
    value=0.0009,
    calibration_dataset="MATH-500",
    calibration_method=(
        "Mean per-step KL divergence on evaluation items where "
        "base model and baseline GRPO model diverged in correctness."
    ),
)

ENTROPY_THRESHOLD = ThresholdConfig(
    signal_type="entropy",
    value=0.075,
    calibration_dataset="MATH-500",
    calibration_method=(
        "Mean per-step Shannon entropy on evaluation items where "
        "base model and baseline GRPO model diverged in correctness."
    ),
)
