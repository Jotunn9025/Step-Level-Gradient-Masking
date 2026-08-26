"""Masking signals — per-step metrics that drive masking decisions.

A MaskingSignal computes a scalar score for each reasoning step.
This score is then fed to a MaskingFunction to produce the binary mask.

Available signals:
  - KLDivergenceSignal: Mean KL(ref || policy) per step
  - EntropySignal: Mean Shannon entropy per step

To add a new signal, subclass MaskingSignal and implement compute_per_step().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

import torch


class MaskingSignal(ABC):
    """Base class for step-level masking signals.

    A signal computes a per-step scalar that quantifies some property
    of the model's behavior on that step (e.g., divergence, uncertainty).

    Subclasses must set:
        name: Human-readable signal name (used in logging).
        required_inputs: List of kwarg keys this signal needs. The trainer
            uses this to generically select the right tensors from the
            training batch — no isinstance checks needed.
    """

    name: str = "base"
    required_inputs: List[str] = []

    @abstractmethod
    def compute_per_step(
        self,
        step_boundaries: List[List[Tuple[int, int]]],
        mask: torch.Tensor,
        **kwargs,
    ) -> List[List[float]]:
        """Compute the signal value for each step in each batch sample.

        Args:
            step_boundaries: Per-sample list of (start_tok, end_tok) ranges.
            mask: (B, T) completion mask.
            **kwargs: Signal-specific tensors (logps, entropies, etc.).

        Returns:
            Per-sample list of per-step float values.
        """
        ...


class KLDivergenceSignal(MaskingSignal):
    """Per-step mean KL divergence between reference and policy.

    Measures how much the policy has drifted from the reference model
    at each reasoning step. High KL → model has diverged (novel reasoning).
    Low KL → model behaves identically to reference (redundant step).

    Required kwargs:
        ref_per_token_logps: (B, T) reference model log-probabilities
        policy_per_token_logps: (B, T) policy model log-probabilities
    """

    name: str = "kl_divergence"
    required_inputs: List[str] = ["ref_per_token_logps", "policy_per_token_logps"]

    def compute_per_step(
        self,
        step_boundaries: List[List[Tuple[int, int]]],
        mask: torch.Tensor,
        **kwargs,
    ) -> List[List[float]]:
        ref_logps = kwargs["ref_per_token_logps"]
        policy_logps = kwargs["policy_per_token_logps"]

        batch_kls = []
        for b, boundaries in enumerate(step_boundaries):
            sample_kls = []
            for (s, e) in boundaries:
                step_mask = mask[b, s:e]
                if step_mask.sum() < 1:
                    sample_kls.append(0.0)
                    continue
                log_ratio = ref_logps[b, s:e] - policy_logps[b, s:e]
                kl_tokens = torch.exp(log_ratio) - log_ratio - 1
                step_kl = (kl_tokens * step_mask).sum() / step_mask.sum().clamp(min=1.0)
                sample_kls.append(step_kl.item())
            batch_kls.append(sample_kls)
        return batch_kls


class EntropySignal(MaskingSignal):
    """Per-step mean Shannon entropy of the policy distribution.

    Measures the model's internal uncertainty at each reasoning step.
    High entropy → model is uncertain (decision boundary).
    Low entropy → model is confident (already learned).

    Required kwargs:
        entropies: (B, T) per-token Shannon entropy values
    """

    name: str = "entropy"
    required_inputs: List[str] = ["entropies"]

    def compute_per_step(
        self,
        step_boundaries: List[List[Tuple[int, int]]],
        mask: torch.Tensor,
        **kwargs,
    ) -> List[List[float]]:
        entropies = kwargs["entropies"]

        batch_entropies = []
        for b, boundaries in enumerate(step_boundaries):
            sample_entropies = []
            for (s, e) in boundaries:
                step_mask = mask[b, s:e]
                if step_mask.sum() < 1:
                    sample_entropies.append(0.0)
                    continue
                step_ent = (entropies[b, s:e] * step_mask).sum() / step_mask.sum().clamp(min=1.0)
                sample_entropies.append(step_ent.item())
            batch_entropies.append(sample_entropies)
        return batch_entropies
