"""Baseline GRPO configuration — no step masking.

Standard Group Relative Policy Optimization with no per-step
loss masking. All generated tokens receive gradient updates.
Used as the control condition for comparison.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BaselineConfig:
    """Baseline GRPO training configuration."""

    # ── Model ──
    model_name: str = "Qwen/Qwen3-4B"

    # ── LoRA ──
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # ── GRPO ──
    k: int = 4                      # Group size (num generations per prompt)
    kl_coeff: float = 0.01          # KL penalty coefficient (beta)
    clip_eps: float = 0.2           # PPO clip epsilon
    lr: float = 1e-5
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    max_grad_norm: float = 1.0
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 2

    # ── Step masking ──
    masking_signal: Optional[str] = None       # No masking for baseline
    masking_threshold: Optional[float] = None
    min_steps_for_masking: int = 2

    # ── vLLM ──
    use_vllm: bool = True
    vllm_gpu_memory_utilization: float = 0.5
    vllm_mode: str = "colocate"
    vllm_enable_sleep_mode: bool = False

    # ── Data ──
    train_data_path: str = "data/rlvr_math_stratified_2k.json"
    eval_data_path: str = "data/eval.json"

    # ── Training ──
    epochs: int = 1
    resume_from_step: int = 0

    # ── Output ──
    checkpoint_every: int = 200
    checkpoint_dir: str = "checkpoints_baseline"
    weights_dir: str = "baseline_weights"
    log_dir: str = "logs_baseline"
