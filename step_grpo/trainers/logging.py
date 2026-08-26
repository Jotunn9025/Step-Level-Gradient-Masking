"""Training logger — CSV step logging + run metadata.

Provides a unified logger that works for all training modes
(baseline, KL-masked, entropy-masked) by dynamically including
signal-specific fields based on the masking configuration.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# Base fields logged for every training step (k-independent)
_BASE_FIELDS = [
    "step", "epoch", "question_idx", "question_text", "ground_truth",
]

# Per-completion fields — expanded for each of the k completions
_PER_COMPLETION_FIELDS = [
    "reward", "advantage", "completion", "extracted_answer",
    "correct", "has_think", "has_steps", "n_steps", "output_tokens",
]

# Aggregate / training fields
_AGGREGATE_FIELDS = [
    "mean_reward", "max_reward", "min_reward", "reward_std",
    "all_zero_advantage",
    "grpo_loss", "grad_norm", "lr",
    "mean_kl", "mean_importance_ratio", "clip_fraction",
    # Step masking metrics (generic)
    "mask_frac_masked", "mask_threshold", "mask_mean_step_score", "mask_num_steps",
    "gpu_mem_alloc_gb", "gpu_mem_reserved_gb",
    "step_wall_time_s", "timestamp",
]


def build_step_fields(k: int = 4) -> list:
    """Build the full list of CSV fields for a given group size k."""
    fields = list(_BASE_FIELDS)
    for i in range(1, k + 1):
        for f in _PER_COMPLETION_FIELDS:
            fields.append(f"{f}_{i}")
    fields.extend(_AGGREGATE_FIELDS)
    return fields


class TrainingLogger:
    """CSV + JSON logger for Step-GRPO training runs.

    Args:
        log_dir: Directory to write log files into.
        log_filename: Name of the CSV log file.
        k: Number of completions per prompt (group size).
        extra_fields: Additional CSV fields for signal-specific metrics.
    """

    def __init__(
        self,
        log_dir: str,
        log_filename: str = "training_log.csv",
        k: int = 4,
        extra_fields: Optional[list] = None,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        fields = build_step_fields(k)
        if extra_fields:
            fields.extend(extra_fields)

        self._f = open(self.log_dir / log_filename, "w", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=fields, extrasaction="ignore")
        self._w.writeheader()

    def log_step(self, row: dict):
        """Write a single training step to the CSV log."""
        self._w.writerow(row)
        self._f.flush()

    def write_meta(self, meta: dict):
        """Write run metadata to a JSON file."""
        with open(self.log_dir / "run_meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

    def close(self):
        """Close the CSV file handle."""
        self._f.close()
