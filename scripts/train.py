"""Unified training entry point for Step-GRPO.

Supports all three training modes via --mode flag:
  - baseline: Standard GRPO (no masking)
  - kl:       SA-GRPO (KL divergence masking)
  - entropy:  EA-GRPO (Entropy masking)

Usage:
  python scripts/train.py --mode baseline
  python scripts/train.py --mode kl
  python scripts/train.py --mode entropy
  python scripts/train.py --mode entropy --threshold 0.1  # override threshold
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig as TRLGRPOConfig

# Add parent directory to path so step_grpo is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from step_grpo.trainers.base_step_grpo import StepGRPOTrainer
from step_grpo.trainers.logging import TrainingLogger
from step_grpo.masking.signals import KLDivergenceSignal, EntropySignal
from step_grpo.masking.functions import HeavisideStepMask
from step_grpo.rewards.math_reward import math_reward_fn
from step_grpo.evaluation.answer_matching import extract_answer, answers_match

# Regex for step counting in logging
_STEP_TAG_RE = re.compile(r"<step>", re.DOTALL)


# ──────────────────────────────────────────────────────────────
# Dataset loading
# ──────────────────────────────────────────────────────────────

def load_local_dataset(path: str) -> Dataset:
    """Load a JSON dataset and format it for GRPO training."""
    with open(path) as f:
        data = json.load(f)

    from step_grpo.steps.strategy import PromptedStepStrategy
    strategy = PromptedStepStrategy()

    formatted = []
    for row in data:
        prompt = strategy.get_prompt_template(row["question"])
        formatted.append({"prompt": prompt, "ground_truth": row["ground_truth"]})
    return Dataset.from_list(formatted)


# ──────────────────────────────────────────────────────────────
# Config loading
# ──────────────────────────────────────────────────────────────

def load_config(mode: str, threshold_override: float = None):
    """Load the appropriate config for the training mode."""
    if mode == "baseline":
        from configs.baseline import BaselineConfig
        config = BaselineConfig()
    elif mode == "kl":
        from configs.kl_masking import KLMaskingConfig
        config = KLMaskingConfig()
    elif mode == "entropy":
        from configs.entropy_masking import EntropyMaskingConfig
        config = EntropyMaskingConfig()
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'baseline', 'kl', or 'entropy'.")

    if threshold_override is not None and config.masking_threshold is not None:
        config.masking_threshold = threshold_override

    return config


def build_masking_components(config):
    """Build the masking signal and function from config."""
    if config.masking_signal is None:
        return None, None

    if config.masking_signal == "kl_divergence":
        signal = KLDivergenceSignal()
    elif config.masking_signal == "entropy":
        signal = EntropySignal()
    else:
        raise ValueError(f"Unknown masking signal: {config.masking_signal}")

    function = HeavisideStepMask(
        threshold=config.masking_threshold,
        min_steps=config.min_steps_for_masking,
    )

    return signal, function


# ──────────────────────────────────────────────────────────────
# Logging callback
# ──────────────────────────────────────────────────────────────

class StepLoggingCallback(TrainerCallback):
    """Callback that logs per-step metrics to the TrainingLogger."""

    def __init__(self, logger: TrainingLogger, config, shared_state: dict, tokenizer):
        self.logger = logger
        self.config = config
        self.shared_state = shared_state
        self.tokenizer = tokenizer
        self._step_start = time.time()

    def on_step_begin(self, args, state, control, **kwargs):
        self._step_start = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or state.global_step == 0:
            return

        step = state.global_step
        elapsed = time.time() - self._step_start
        gpu_alloc = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        gpu_res = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0

        row = {
            "step": step,
            "epoch": int(state.epoch or 0),
            "grpo_loss": round(logs.get("loss", 0.0), 6),
            "grad_norm": round(logs.get("grad_norm", 0.0), 6),
            "lr": logs.get("learning_rate", self.config.lr),
            "mean_kl": round(logs.get("objective/kl", logs.get("kl", 0.0)), 6),
            "gpu_mem_alloc_gb": round(gpu_alloc, 3),
            "gpu_mem_reserved_gb": round(gpu_res, 3),
            "step_wall_time_s": round(elapsed, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mean_reward": round(logs.get("rewards/mean", logs.get("reward", 0.0)), 4),
            "reward_std": round(logs.get("rewards/std", 0.0), 4),
        }

        # Masking-specific metrics
        if self.config.masking_signal == "kl_divergence":
            row["mask_frac_masked"] = round(logs.get("step_mask/kl_divergence/frac_tokens_masked", 0.0), 4)
            row["mask_mean_step_score"] = round(logs.get("step_mask/kl_divergence/mean_step_score", 0.0), 6)
            row["mask_num_steps"] = round(logs.get("step_mask/kl_divergence/num_steps_found", 0.0), 2)
            row["mask_threshold"] = self.config.masking_threshold
        elif self.config.masking_signal == "entropy":
            row["mask_frac_masked"] = round(logs.get("step_mask/entropy/frac_tokens_masked", 0.0), 4)
            row["mask_mean_step_score"] = round(logs.get("step_mask/entropy/mean_step_score", 0.0), 6)
            row["mask_num_steps"] = round(logs.get("step_mask/entropy/num_steps_found", 0.0), 2)
            row["mask_threshold"] = self.config.masking_threshold

        # Log completions and rewards
        prompts = self.shared_state.get("prompts", [])
        completions = self.shared_state.get("completions", [])
        gts = self.shared_state.get("ground_truths", [])
        rewards = self.shared_state.get("rewards", [])
        self.shared_state.update({"prompts": [], "completions": [], "ground_truths": [], "rewards": []})

        if prompts:
            row["question_text"] = prompts[0]
        if gts:
            row["ground_truth"] = gts[0] if isinstance(gts, list) else gts

        if len(rewards) > 1:
            r_arr = np.array(rewards)
            std_val = r_arr.std()
            row["all_zero_advantage"] = 1 if std_val < 1e-6 else 0
            for i in range(min(self.config.k, len(rewards))):
                row[f"advantage_{i+1}"] = round(
                    (rewards[i] - r_arr.mean()) / (std_val + 1e-8), 4
                )

        for i in range(self.config.k):
            if i < len(completions):
                comp = completions[i]
                row[f"completion_{i+1}"] = comp
                row[f"has_think_{i+1}"] = 1 if ("<think>" in comp and "</think>" in comp) else 0
                row[f"has_steps_{i+1}"] = 1 if "<step>" in comp else 0
                row[f"n_steps_{i+1}"] = len(_STEP_TAG_RE.findall(comp))
                ext = extract_answer(comp)
                row[f"extracted_answer_{i+1}"] = ext
                if gts:
                    gt = gts[i] if i < len(gts) else (gts[0] if isinstance(gts, list) else gts)
                    row[f"correct_{i+1}"] = 1 if answers_match(ext, gt) else 0
                try:
                    row[f"output_tokens_{i+1}"] = len(self.tokenizer.encode(comp))
                except Exception:
                    row[f"output_tokens_{i+1}"] = len(comp) // 4
            if i < len(rewards):
                row[f"reward_{i+1}"] = rewards[i]

        if rewards:
            row["max_reward"] = max(rewards)
            row["min_reward"] = min(rewards)

        self.logger.log_step(row)

        mask_str = ""
        if self.config.masking_signal:
            mask_str = f" masked={row.get('mask_frac_masked', 0):.2%}"

        print(
            f"[Step {step}] reward={row['mean_reward']:.4f} "
            f"loss={row['grpo_loss']:.6f} kl={row['mean_kl']:.6f}"
            f"{mask_str} gpu={gpu_alloc:.1f}GB t={elapsed:.1f}s",
            flush=True,
        )


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Step-GRPO Training")
    parser.add_argument(
        "--mode", required=True, choices=["baseline", "kl", "entropy"],
        help="Training mode: baseline (no masking), kl (SA-GRPO), entropy (EA-GRPO)",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Override the default masking threshold.",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override the model name/path (default: from config).",
    )
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Override training data path (default: from config).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override base output directory for checkpoints, weights, and logs.",
    )
    args = parser.parse_args()

    # ── Load config ──
    config = load_config(args.mode, args.threshold)

    # Apply CLI overrides
    if args.model:
        config.model_name = args.model
    if args.data_path:
        config.train_data_path = args.data_path
    if args.output_dir:
        base = args.output_dir.rstrip("/")
        config.checkpoint_dir = f"{base}/checkpoints"
        config.weights_dir = f"{base}/weights"
        config.log_dir = f"{base}/logs"

    print("Config loaded:")
    for k, v in asdict(config).items():
        print(f"  {k}: {v}")

    # ── GPU check ──
    print(f"\nPyTorch {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: No GPU detected!")

    # ── Load dataset ──
    train_ds = load_local_dataset(config.train_data_path)
    print(f"\nTraining dataset: {len(train_ds)} examples")

    # ── Load model + LoRA ──
    print(f"\nLoading tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {config.model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name, torch_dtype=torch.bfloat16, device_map="auto",
    )
    for p in base_model.parameters():
        p.requires_grad = False

    lora_cfg = LoraConfig(
        r=config.lora_r, lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.enable_input_require_grads()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")

    # ── Build masking components ──
    masking_signal, masking_function = build_masking_components(config)

    # ── Setup logging ──
    logger = TrainingLogger(config.log_dir, log_filename="training_log.csv", k=config.k)
    run_id = f"step-grpo-{args.mode}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    logger.write_meta({
        "run_id": run_id,
        "mode": args.mode,
        "model": config.model_name,
        "n_questions": len(train_ds),
        "config": asdict(config),
        "trainable_params": trainable,
        "total_params": total,
        "gpu": gpu_name,
        "gpu_vram_gb": round(gpu_mem, 2),
        "torch_version": torch.__version__,
    })

    # ── Shared state for reward logging ──
    shared_state = {"prompts": [], "completions": [], "ground_truths": [], "rewards": []}

    def _reward_fn(prompts, completions, **kwargs):
        ground_truths = kwargs.get("ground_truth", [])
        rewards = math_reward_fn(prompts, completions, ground_truths)
        shared_state["prompts"] = prompts
        shared_state["completions"] = completions
        shared_state["ground_truths"] = ground_truths
        shared_state["rewards"] = rewards
        return rewards

    # ── Build TRL GRPOConfig ──
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.weights_dir).mkdir(parents=True, exist_ok=True)

    grpo_kwargs = dict(
        num_generations=config.k, beta=config.kl_coeff,
        epsilon=config.clip_eps, learning_rate=config.lr,
        max_completion_length=config.max_new_tokens,
        temperature=config.temperature, top_p=config.top_p,
        max_grad_norm=config.max_grad_norm,
        num_train_epochs=config.epochs,
        save_steps=config.checkpoint_every, save_total_limit=3,
        output_dir=config.checkpoint_dir, report_to="none",
        run_name=run_id, logging_steps=1,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=0,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    if config.use_vllm:
        grpo_kwargs["use_vllm"] = True
        grpo_kwargs["vllm_gpu_memory_utilization"] = config.vllm_gpu_memory_utilization
        grpo_kwargs["vllm_mode"] = config.vllm_mode
        grpo_kwargs["vllm_enable_sleep_mode"] = config.vllm_enable_sleep_mode

    trl_cfg = TRLGRPOConfig(**grpo_kwargs)

    # ── Build trainer ──
    callback = StepLoggingCallback(logger, config, shared_state, tokenizer)

    trainer_kwargs = {
        "model": model,
        "reward_funcs": [_reward_fn],
        "args": trl_cfg,
        "train_dataset": train_ds,
        "callbacks": [callback],
        "masking_signal": masking_signal,
        "masking_function": masking_function,
    }

    # Handle tokenizer/processing_class API difference
    from trl import GRPOTrainer as _Base
    if "processing_class" in inspect.signature(_Base.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = StepGRPOTrainer(**trainer_kwargs)

    # ── Train ──
    mode_names = {"baseline": "BASELINE GRPO", "kl": "SA-GRPO (KL)", "entropy": "EA-GRPO (ENTROPY)"}
    print(f"\n{'='*60}")
    print(f"  {mode_names[args.mode]} TRAINING")
    print(f"  Run: {run_id}")
    print(f"  Dataset: {len(train_ds)} questions, k={config.k}")
    if config.masking_threshold is not None:
        print(f"  Masking: {config.masking_signal}, threshold={config.masking_threshold}")
    print(f"{'='*60}\n")

    resume_ckpt = None
    if config.resume_from_step > 0:
        resume_ckpt = os.path.join(config.checkpoint_dir, f"checkpoint-{config.resume_from_step}")

    try:
        train_result = trainer.train(resume_from_checkpoint=resume_ckpt)
        trainer.save_model(config.weights_dir)
        logger.close()
        print(f"\nDone! Adapter → {config.weights_dir}, Logs → {config.log_dir}/training_log.csv")
    except Exception as e:
        print(e)
        if os.environ.get("RUNPOD_POD_ID"):
            os.system("runpodctl stop pod")


if __name__ == "__main__":
    main()
