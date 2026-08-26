"""Threshold calibration for step-level masking.

Computes the optimal masking threshold by analyzing divergent items
between a base model and a fine-tuned baseline model. The threshold
is the average per-step signal score on items where the two models
disagree on correctness.

Supports both KL divergence and entropy signals.

Usage:
  python scripts/calibrate_threshold.py \\
    --signal entropy \\
    --base_csv logs/base_model_eval.csv \\
    --baseline_csv logs/baseline_eval.csv \\
    --lora_path checkpoints_baseline/checkpoint-2000

  python scripts/calibrate_threshold.py \\
    --signal kl \\
    --base_csv logs/base_model_eval.csv \\
    --baseline_csv logs/baseline_eval.csv \\
    --lora_path checkpoints_baseline/checkpoint-2000
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from step_grpo.steps.boundary_detector import find_step_boundaries


MODEL_NAME = "Qwen/Qwen3-4B"


def load_eval_csv(path: str) -> list:
    """Load evaluation CSV into list of dicts."""
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_divergent_items(base_rows, baseline_rows):
    """Find items where base and baseline models disagree on correctness."""
    divergent = []
    for b_row, bl_row in zip(base_rows, baseline_rows):
        b_correct = int(b_row.get("correct", 0))
        bl_correct = int(bl_row.get("correct", 0))
        if b_correct != bl_correct:
            divergent.append({
                "idx": int(b_row["idx"]),
                "base_correct": b_correct,
                "baseline_correct": bl_correct,
                "base_response": b_row.get("model_response", ""),
                "baseline_response": bl_row.get("model_response", ""),
                "question": b_row.get("question", ""),
                "ground_truth": b_row.get("ground_truth", ""),
            })
    return divergent


def compute_step_scores(
    model, tokenizer, responses: list, signal_type: str,
    ref_model=None,
) -> list:
    """Compute per-step signal scores for a list of responses."""
    all_step_scores = []

    for response in responses:
        tokens = tokenizer(response, return_tensors="pt", add_special_tokens=False)
        input_ids = tokens["input_ids"].to(model.device)
        attention_mask = tokens["attention_mask"].to(model.device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        # Get per-token log-probabilities
        log_probs = torch.nn.functional.log_softmax(logits[:, :-1], dim=-1)
        target_ids = input_ids[:, 1:]
        token_logps = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

        # Completion mask (all tokens are completion in this context)
        comp_mask = attention_mask[:, 1:].float()

        # Find step boundaries
        boundaries = find_step_boundaries(target_ids, comp_mask.int(), tokenizer)

        if signal_type == "entropy":
            # Compute per-token entropy
            probs = torch.softmax(logits[:, :-1], dim=-1)
            entropies = -(probs * (probs + 1e-10).log()).sum(-1)

            # Mean entropy per step
            for b_bounds in boundaries:
                step_scores = []
                for (s, e) in b_bounds:
                    m = comp_mask[0, s:e]
                    if m.sum() < 1:
                        step_scores.append(0.0)
                        continue
                    step_ent = (entropies[0, s:e] * m).sum() / m.sum()
                    step_scores.append(step_ent.item())
                all_step_scores.extend(step_scores)

        elif signal_type == "kl" and ref_model is not None:
            # Get reference model log-probabilities
            with torch.no_grad():
                ref_outputs = ref_model(input_ids, attention_mask=attention_mask)
                ref_log_probs = torch.nn.functional.log_softmax(ref_outputs.logits[:, :-1], dim=-1)
                ref_token_logps = ref_log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

            # Mean KL per step
            for b_bounds in boundaries:
                step_scores = []
                for (s, e) in b_bounds:
                    m = comp_mask[0, s:e]
                    if m.sum() < 1:
                        step_scores.append(0.0)
                        continue
                    log_ratio = ref_token_logps[0, s:e] - token_logps[0, s:e]
                    kl_tokens = torch.exp(log_ratio) - log_ratio - 1
                    step_kl = (kl_tokens * m).sum() / m.sum()
                    step_scores.append(step_kl.item())
                all_step_scores.extend(step_scores)

    return all_step_scores


def main():
    parser = argparse.ArgumentParser(description="Calibrate masking threshold")
    parser.add_argument("--signal", required=True, choices=["kl", "entropy"])
    parser.add_argument("--base_csv", required=True, help="Base model eval CSV")
    parser.add_argument("--baseline_csv", required=True, help="Baseline model eval CSV")
    parser.add_argument("--lora_path", required=True, help="Path to baseline LoRA adapter")
    parser.add_argument("--model_name", default=MODEL_NAME)
    args = parser.parse_args()

    # Load eval results
    base_rows = load_eval_csv(args.base_csv)
    baseline_rows = load_eval_csv(args.baseline_csv)
    divergent = find_divergent_items(base_rows, baseline_rows)
    print(f"Found {len(divergent)} divergent items out of {len(base_rows)} total")

    if not divergent:
        print("No divergent items found. Cannot calibrate threshold.")
        return

    # Load models
    print(f"\nLoading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print(f"Loading base model: {args.model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto",
    )

    print(f"Loading LoRA adapter: {args.lora_path}")
    lora_model = PeftModel.from_pretrained(base_model, args.lora_path)

    # Compute scores on divergent item responses
    responses = [item["baseline_response"] for item in divergent]
    ref_model = base_model if args.signal == "kl" else None

    print(f"\nComputing per-step {args.signal} scores on {len(responses)} divergent responses...")
    scores = compute_step_scores(
        lora_model, tokenizer, responses, args.signal, ref_model=ref_model,
    )

    if not scores:
        print("No step scores computed. Check if responses have <step> tags.")
        return

    import numpy as np
    scores_arr = np.array(scores)
    threshold = float(scores_arr.mean())

    print(f"\n{'='*60}")
    print(f"  CALIBRATION RESULTS")
    print(f"  Signal: {args.signal}")
    print(f"  Divergent items: {len(divergent)}")
    print(f"  Total steps analyzed: {len(scores)}")
    print(f"  Mean score: {threshold:.6f}")
    print(f"  Std:  {scores_arr.std():.6f}")
    print(f"  Min:  {scores_arr.min():.6f}")
    print(f"  Max:  {scores_arr.max():.6f}")
    print(f"")
    print(f"  >>> RECOMMENDED THRESHOLD: {threshold:.6f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
