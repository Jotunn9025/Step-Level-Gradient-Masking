"""Multi-run comparative evaluation across models and datasets.

Evaluates Base, Baseline GRPO, SA-GRPO, and EA-GRPO models on
MATH-500, GSM8K, and AIME using stochastic multi-pass inference
for statistical robustness.

Usage:
  python scripts/evaluate.py --dataset math500 --n-runs 25 --output-dir logs_eval
  python scripts/evaluate.py --dataset gsm8k --n-runs 10 --output-dir logs_eval_gsm8k
  python scripts/evaluate.py --dataset aime --n-runs 20 --output-dir logs_eval_aime --max-tokens 4096
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from step_grpo.evaluation.answer_matching import (
    extract_answer, answers_match, answers_match_numeric,
)
from step_grpo.steps.strategy import PromptedStepStrategy

import re
_STEP_TAG_RE = re.compile(r"<step>", re.DOTALL)


# ──────────────────────────────────────────────────────────────
# Defaults (overridable via CLI args)
# ──────────────────────────────────────────────────────────────
DEFAULT_MODEL = "Qwen/Qwen3-4B"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_GPU_MEM_UTIL = 0.7

DATASET_PATHS = {
    "math500": "data/eval.json",
    "gsm8k": "data/eval_gsm8k.json",
    "aime": "data/eval_aime.json",
}

FIELDNAMES = [
    "idx", "question", "ground_truth",
    "model_response", "extracted_answer", "correct",
    "has_think", "has_answer_tag", "n_steps",
    "response_length_chars", "response_tokens", "timestamp",
]


def evaluate_outputs(eval_data, outputs, csv_path, match_fn=answers_match):
    """Evaluate model outputs, write CSV, return accuracy %."""
    correct = 0
    total = len(eval_data)
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for i, (row, output) in enumerate(zip(eval_data, outputs)):
            response = output.outputs[0].text
            gt = row["ground_truth"]
            extracted = extract_answer(response)
            is_correct = match_fn(extracted, gt) if extracted else False
            if is_correct:
                correct += 1
            writer.writerow({
                "idx": i, "question": row["question"], "ground_truth": gt,
                "model_response": response, "extracted_answer": extracted,
                "correct": 1 if is_correct else 0,
                "has_think": 1 if ("<think>" in response and "</think>" in response) else 0,
                "has_answer_tag": 1 if ("<answer>" in response and "</answer>" in response) else 0,
                "n_steps": len(_STEP_TAG_RE.findall(response)),
                "response_length_chars": len(response),
                "response_tokens": len(output.outputs[0].token_ids),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
    return correct / total * 100


def main():
    parser = argparse.ArgumentParser(description="Multi-run comparative evaluation")
    parser.add_argument("--dataset", required=True, choices=["math500", "gsm8k", "aime"])
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Base model name/path (default: {DEFAULT_MODEL})")
    parser.add_argument("--gpu-mem-util", type=float, default=DEFAULT_GPU_MEM_UTIL,
                        help=f"vLLM GPU memory utilization (default: {DEFAULT_GPU_MEM_UTIL})")
    parser.add_argument("--baseline-checkpoint", default="checkpoints_baseline/checkpoint-2000")
    parser.add_argument("--sagrpo-checkpoint", default="checkpoints_sa_grpo/checkpoint-2000")
    parser.add_argument("--eagrpo-checkpoint", default="checkpoints_ea_grpo/checkpoint-2000")
    args = parser.parse_args()

    data_path = DATASET_PATHS[args.dataset]
    if not Path(data_path).exists():
        print(f"ERROR: {data_path} not found. Run scripts/prepare_data.py first.")
        return

    match_fn = answers_match_numeric if args.dataset in ("gsm8k", "aime") else answers_match
    strategy = PromptedStepStrategy()

    with open(data_path) as f:
        eval_data = json.load(f)
    prompts = [strategy.get_prompt_template(row["question"]) for row in eval_data]
    print(f"Eval problems: {len(eval_data)}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gc.collect()
    torch.cuda.empty_cache()

    llm_kwargs = {
        "model": args.model, "dtype": "bfloat16",
        "enable_lora": True, "max_lora_rank": 64,
        "gpu_memory_utilization": args.gpu_mem_util,
    }
    if args.max_model_len:
        llm_kwargs["max_model_len"] = args.max_model_len

    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(temperature=DEFAULT_TEMPERATURE, top_p=DEFAULT_TOP_P, max_tokens=args.max_tokens)

    lora_requests = {
        "baseline": LoRARequest("baseline_adapter", 1, args.baseline_checkpoint),
        "sagrpo": LoRARequest("sagrpo_adapter", 2, args.sagrpo_checkpoint),
        "eagrpo": LoRARequest("eagrpo_adapter", 3, args.eagrpo_checkpoint),
    }

    all_accs = {"base": [], "baseline": [], "sagrpo": [], "eagrpo": []}
    N = args.n_runs
    total_start = time.time()

    print(f"\n{'='*60}")
    print(f"  COMPARATIVE EVAL: {args.dataset.upper()}")
    print(f"  {len(eval_data)} problems × {N} runs × 4 models")
    print(f"{'='*60}\n")

    for run in tqdm(range(1, N + 1), desc="Eval runs", unit="run"):
        tqdm.write(f"\n--- Run {run}/{N} ---")

        # Base model
        out = llm.generate(prompts, sampling_params)
        acc = evaluate_outputs(eval_data, out, out_dir / f"base_model_run{run}.csv", match_fn)
        all_accs["base"].append(acc)
        tqdm.write(f"  Base: {acc:.1f}%")
        del out

        # Baseline GRPO
        out = llm.generate(prompts, sampling_params, lora_request=lora_requests["baseline"])
        acc = evaluate_outputs(eval_data, out, out_dir / f"baseline_grpo_run{run}.csv", match_fn)
        all_accs["baseline"].append(acc)
        tqdm.write(f"  Baseline: {acc:.1f}%")
        del out

        # SA-GRPO
        out = llm.generate(prompts, sampling_params, lora_request=lora_requests["sagrpo"])
        acc = evaluate_outputs(eval_data, out, out_dir / f"sagrpo_run{run}.csv", match_fn)
        all_accs["sagrpo"].append(acc)
        tqdm.write(f"  SA-GRPO: {acc:.1f}%")
        del out

        # EA-GRPO
        out = llm.generate(prompts, sampling_params, lora_request=lora_requests["eagrpo"])
        acc = evaluate_outputs(eval_data, out, out_dir / f"eagrpo_run{run}.csv", match_fn)
        all_accs["eagrpo"].append(acc)
        tqdm.write(f"  EA-GRPO: {acc:.1f}%")
        del out

        gc.collect()

    total_time = time.time() - total_start

    def compute_stats(accs):
        arr = np.array(accs)
        return {"mean": round(float(arr.mean()), 2), "std": round(float(arr.std()), 2),
                "min": round(float(arr.min()), 2), "max": round(float(arr.max()), 2),
                "runs": [round(float(a), 1) for a in arr]}

    stats = {k: compute_stats(v) for k, v in all_accs.items()}

    # Save summary
    summary = {
        "eval_timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset, "model": args.model, "n_runs": N,
        "n_problems": len(eval_data),
        "total_wall_time_min": round(total_time / 60, 2),
        **{k: v for k, v in stats.items()},
    }
    with open(out_dir / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS — {args.dataset.upper()}")
    print(f"{'='*60}")
    for name, label in [("base", "Base"), ("baseline", "Baseline"), ("sagrpo", "SA-GRPO"), ("eagrpo", "EA-GRPO")]:
        s = stats[name]
        print(f"  {label:12s}: {s['mean']:.2f}% ± {s['std']:.2f}")
    print(f"  Total time: {total_time/60:.1f} min")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
