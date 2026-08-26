"""Masking functions — convert per-step signal scores into binary token masks.

A MaskingFunction takes per-step scalar scores from a MaskingSignal
and produces a (B, T) binary mask tensor. This mask is multiplied
into the per-token loss to zero out steps that should not receive
gradient updates.

Available functions:
  - HeavisideStepMask: Binary step function — find first step exceeding
    threshold, mask everything before it.

To add a new masking function (e.g., exponential decay, soft weighting),
subclass MaskingFunction and implement build_mask().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch


class MaskingFunction(ABC):
    """Base class for masking functions.

    Converts per-step signal scores into a per-token binary mask.
    """

    name: str = "base"

    @abstractmethod
    def build_mask(
        self,
        per_step_scores: List[List[float]],
        step_boundaries: List[List[Tuple[int, int]]],
        advantages: torch.Tensor,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build a (B, T) binary mask from per-step scores.

        Args:
            per_step_scores: Per-sample, per-step scalar values from a MaskingSignal.
            step_boundaries: Per-sample list of (start_tok, end_tok) ranges.
            advantages: (B,) or (B, 1) advantage values.
            seq_len: Sequence length T.
            device: Target device.

        Returns:
            (B, T) tensor of 0s and 1s. 0 = masked (no gradient), 1 = active.
        """
        ...


class HeavisideStepMask(MaskingFunction):
    """Heaviside (step function) masking.

    Walks steps in order and finds the FIRST step where the signal
    exceeds the threshold (the "branching point" / "uncertainty point").
    All tokens before that step are masked (set to 0). All tokens from
    that step onward are kept (set to 1).

    This implements the core insight: early confident steps are redundant
    for RL. Training should focus on the exact point where the model
    begins to diverge/struggle.

    Edge cases (all result in all-1s mask = standard GRPO fallback):
      - advantage ≈ 0: no loss anyway
      - fewer than min_steps: too few steps to meaningfully mask
      - all scores below threshold: model is confident everywhere
      - first step already above threshold: uncertainty from the start

    Args:
        threshold: Signal value above which a step is considered
            "divergent" (KL) or "uncertain" (entropy).
        min_steps: Minimum number of detected steps required before
            masking is applied. Default: 2.
    """

    name: str = "heaviside"

    def __init__(self, threshold: float, min_steps: int = 2):
        self.threshold = threshold
        self.min_steps = min_steps

    def build_mask(
        self,
        per_step_scores: List[List[float]],
        step_boundaries: List[List[Tuple[int, int]]],
        advantages: torch.Tensor,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        batch_size = len(per_step_scores)
        step_mask = torch.ones(batch_size, seq_len, device=device)

        for b in range(batch_size):
            scores = per_step_scores[b]
            bounds = step_boundaries[b]

            # Skip if: no advantage, too few steps, or no steps detected
            adv_val = (
                advantages[b].item()
                if advantages.dim() == 1
                else advantages[b, 0].item()
            )
            if abs(adv_val) < 1e-8 or len(scores) < self.min_steps:
                continue

            # Find first step where score exceeds threshold
            branching_step = None
            for i, score in enumerate(scores):
                if score >= self.threshold:
                    branching_step = i
                    break

            if branching_step is None:
                # All steps below threshold — don't mask everything
                continue

            if branching_step == 0:
                # First step already exceeds threshold — no pre-branching region
                continue

            # Mask all tokens BEFORE the branching step
            mask_end_token = bounds[branching_step][0]
            step_mask[b, :mask_end_token] = 0.0

        return step_mask

    def __repr__(self) -> str:
        return (
            f"HeavisideStepMask(threshold={self.threshold}, "
            f"min_steps={self.min_steps})"
        )
