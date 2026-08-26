"""Step-GRPO Trainer — unified step-level masking for GRPO.

A single GRPOTrainer subclass that supports:
  - Baseline GRPO (no masking): signal=None
  - SA-GRPO (KL masking): signal=KLDivergenceSignal + HeavisideStepMask
  - EA-GRPO (Entropy masking): signal=EntropySignal + HeavisideStepMask
  - Any future combination of signals and masking functions

The trainer overrides only _compute_loss. Generation, scoring, and
advantage computation are inherited from TRL's GRPOTrainer unchanged.

Key design: the masking logic is fully composable. To add a new masking
strategy, implement a MaskingSignal and/or MaskingFunction and pass them
to this trainer. No subclassing needed. The trainer uses each signal's
``required_inputs`` declaration to generically select the right tensors
— it never needs to know about specific signal types.
"""

from __future__ import annotations

from typing import Optional

import torch
from trl import GRPOTrainer

from step_grpo.steps.boundary_detector import find_step_boundaries
from step_grpo.masking.signals import MaskingSignal
from step_grpo.masking.functions import MaskingFunction


class StepGRPOTrainer(GRPOTrainer):
    """GRPOTrainer with composable step-level loss masking.

    Args:
        masking_signal: A MaskingSignal instance that computes per-step
            scores. If None, no masking is applied (baseline GRPO).
        masking_function: A MaskingFunction instance that converts
            per-step scores into a binary token mask. Required if
            masking_signal is not None.
        *args, **kwargs: Forwarded to GRPOTrainer.
    """

    def __init__(
        self,
        *args,
        masking_signal: Optional[MaskingSignal] = None,
        masking_function: Optional[MaskingFunction] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.masking_signal = masking_signal
        self.masking_function = masking_function

        if masking_signal is not None and masking_function is None:
            raise ValueError(
                "masking_function is required when masking_signal is provided."
            )

        if masking_signal is not None:
            print(
                f"[Step-GRPO] Masking enabled: "
                f"signal={masking_signal.name}, "
                f"function={masking_function}"
            )
        else:
            print("[Step-GRPO] Baseline mode (no step masking)")

    def _compute_loss(self, model, inputs):
        """Override parent to inject step-aware masking before loss aggregation.

        The structure follows the parent GRPOTrainer._compute_loss exactly,
        with one addition: after computing per_token_loss and before
        aggregation, we optionally multiply by a step mask.
        """
        # ── 1. Standard forward pass (identical to parent) ──
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        mask = completion_mask if "tool_mask" not in inputs else completion_mask * inputs["tool_mask"]

        per_token_logps, entropies, aux_loss = self._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, logits_to_keep,
            compute_entropy=True,
            compute_aux_loss=self.aux_loss_enabled,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            spatial_shapes=inputs.get("spatial_shapes"),
            num_tiles=inputs.get("num_tiles"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
            mm_token_type_ids=inputs.get("mm_token_type_ids"),
            image_position_ids=inputs.get("image_position_ids"),
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        # ── 2. Advantages & importance sampling (identical to parent) ──
        advantages = inputs["advantages"]
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)

        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        if self.off_policy_mask_threshold is not None:
            sampling_per_token_logps = inputs.get("sampling_per_token_logps", old_per_token_logps)
            off_policy_mask = self.get_off_policy_mask(
                advantages=advantages, per_token_logps=per_token_logps,
                sampling_per_token_logps=sampling_per_token_logps,
                mask=mask, off_policy_threshold=self.off_policy_mask_threshold,
            )

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(f"Unknown importance sampling level: {self.importance_sampling_level}")

        coef_1 = torch.exp(log_importance_weights)

        # ── 3. KL divergence (identical to parent) ──
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps) - 1
            )
            if self.args.use_bias_correction_kl:
                per_token_kl = per_token_kl * coef_1

        # ── 4. Per-token loss (identical to parent, GRPO type only) ──
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)
        per_token_loss1 = coef_1 * advantages
        per_token_loss2 = coef_2 * advantages
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        if self.off_policy_mask_threshold is not None:
            per_token_loss = per_token_loss * off_policy_mask
        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask
        if self.use_vllm and self.vllm_importance_sampling_correction:
            per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        # ══════════════════════════════════════════════════════════
        # ── 5. Step-level masking (the framework's contribution) ──
        # ══════════════════════════════════════════════════════════

        if self.masking_signal is not None:
            mode = "train" if self.model.training else "eval"

            # Detect step boundaries
            step_boundaries = find_step_boundaries(
                completion_ids, completion_mask, self.processing_class,
            )

            # Compute per-step signal scores using generic dispatch.
            # Each signal declares its required_inputs; the trainer maps
            # them from a registry of available tensors. Adding a new
            # signal never requires modifying this trainer.
            available_tensors = {
                "ref_per_token_logps": inputs.get("ref_per_token_logps"),
                "policy_per_token_logps": per_token_logps.detach(),
                "entropies": entropies.detach(),
            }
            signal_kwargs = {
                key: available_tensors[key]
                for key in self.masking_signal.required_inputs
                if available_tensors.get(key) is not None
            }

            per_step_scores = self.masking_signal.compute_per_step(
                step_boundaries, mask, **signal_kwargs,
            )

            # Build the step mask
            step_mask = self.masking_function.build_mask(
                per_step_scores, step_boundaries, advantages,
                seq_len=completion_ids.size(1), device=completion_ids.device,
            )

            # Apply the mask
            per_token_loss = per_token_loss * step_mask

            # Log masking metrics
            all_scores = [s for sample in per_step_scores for s in sample]
            mean_score = sum(all_scores) / max(len(all_scores), 1)
            num_steps = sum(len(b) for b in step_boundaries) / max(len(step_boundaries), 1)
            total_active = mask.sum().item()
            masked_tokens = ((1.0 - step_mask) * mask).sum().item()
            frac_masked = masked_tokens / max(total_active, 1.0)

            prefix = f"step_mask/{self.masking_signal.name}"
            self._metrics[mode][f"{prefix}/frac_tokens_masked"].append(frac_masked)
            self._metrics[mode][f"{prefix}/mean_step_score"].append(mean_score)
            self._metrics[mode][f"{prefix}/num_steps_found"].append(num_steps)

        # ── 6. Aggregate loss (identical to parent, GRPO type) ──
        mode = "train" if self.model.training else "eval"
        loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
        normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
        loss = loss / normalizer

        if self.aux_loss_enabled:
            norm = self.current_gradient_accumulation_steps if mode == "train" else 1.0
            loss = loss + self.router_aux_loss_coef * aux_loss / norm
            self._metrics[mode]["aux_loss"].append(
                self.accelerator.gather_for_metrics(aux_loss).mean().item()
            )

        # ── 7. Standard metrics logging (identical to parent) ──
        completion_token_count = mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:
                return x.mean()
            return (x * mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl_metric = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(
                self.accelerator.gather(mean_kl_metric).nanmean().item()
            )

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(
            self.accelerator.gather(mean_entropy).nanmean().item()
        )

        # Clip ratio metrics
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = masked_batch_mean(is_low_clipped.float())
        high_clip = masked_batch_mean(is_high_clipped.float())
        clip_ratio = masked_batch_mean(is_region_clipped.float())

        gathered_low = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(gathered_low.min().item())
        gathered_high = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(gathered_high.max().item())
        gathered_clip = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip.nanmean().item())

        return loss
