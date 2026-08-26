"""SA-GRPO configuration — KL divergence masking.

Step-Aware GRPO that masks reasoning steps where the policy model
has not diverged from the reference model (KL < threshold).
Focuses RL signal on the branching point where novel reasoning begins.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from step_grpo.masking.thresholds import KL_THRESHOLD


@dataclass
class KLMaskingConfig:
    """SA-GRPO (KL divergence masking) training configuration."""

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
    k: int = 4
    kl_coeff: float = 0.01
    clip_eps: float = 0.2
    lr: float = 1e-5
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    max_grad_norm: float = 1.0
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 2

    # ── Step masking ──
    masking_signal: str = "kl_divergence"
    masking_threshold: float = KL_THRESHOLD.value    # 0.0009
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
    checkpoint_dir: str = "checkpoints_sa_grpo"
    weights_dir: str = "sa_grpo_weights"
    log_dir: str = "logs_sa_grpo"
