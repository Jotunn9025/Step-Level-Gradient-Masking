# Step-GRPO: Step-Level Reinforcement Learning from Verifiable Rewards

A modular framework for **step-level RLVR** that treats the *reasoning step* as the fundamental unit of reinforcement, rather than individual tokens.

> **Hardware**: All training and evaluation was run on a **cloud NVIDIA L40S** GPU with **32 GB VRAM**.

---

## Core Idea

Standard GRPO applies RL gradients uniformly to every token in a model's response. This wastes signal on confident, already-learned reasoning steps and risks overfitting on trivial formatting. **Step-GRPO** detects structured reasoning steps in model outputs, computes a per-step signal (e.g., KL divergence or entropy), and masks out steps where training would be redundant — focusing RL exclusively on the decision boundaries where the model actually needs guidance.

> **Status (v0.1)**: The current implementation ships with *prefix masking* — it blanks all tokens before the first "branching" step and leaves the rest of the response unmasked. A proper **selective step-only** update scheme is planned. Masking thresholds have been validated by the [threshold analysis](#threshold-analysis). See [Roadmap](#roadmap--todo).

---

## Setup

### Option A: Docker (Recommended)

A Dockerfile is included for reproducible setup. It uses the `vishwa123/cuda-13.0-pytorch` base image (PyTorch 2.9 + CUDA 13).

```bash
# Build the image
docker build -t step-grpo .

# Run with GPU access
docker run --runtime=nvidia --gpus all -it step-grpo
```

### Option B: Local / Any Machine

Requires **CUDA 13** and **Python 3.12**.

```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc   # or restart shell

# Create and activate a virtual environment (optional but recommended)
uv venv .venv && source .venv/bin/activate

# Install dependencies
uv pip install peft "trl[vllm]" pandas datasets sympy tqdm 
```

---

## Quick Start

### 1. Prepare Data

```bash
python scripts/prepare_data.py
```

This downloads and formats:
- **GSM8K** test set (1,319 problems)
- **AIME 2025** (30 problems)
- **MATH-500** with subject/difficulty metadata

> The training dataset (`data/rlvr_math_stratified_2k.json`) should already be present. It contains 2,000 stratified math problems for RLVR training.

### 2. Train

All three training modes use a single unified entry point:

```bash
# Baseline GRPO (no masking — control condition)
python scripts/train.py --mode baseline

# SA-GRPO (KL divergence masking)
python scripts/train.py --mode kl

# EA-GRPO (Entropy masking)
python scripts/train.py --mode entropy
```

**Override defaults via CLI** (no need to edit config files):

```bash
# Use a different model
python scripts/train.py --mode entropy --model Qwen/Qwen3-8B

# Use a different dataset
python scripts/train.py --mode baseline --data-path data/my_custom_data.json

# Custom output directory
python scripts/train.py --mode kl --output-dir runs/experiment_1

# Custom masking threshold
python scripts/train.py --mode entropy --threshold 0.1
```

### 3. Calibrate Threshold (Optional)

Default thresholds are provided (calibrated on Qwen3-4B + MATH-500). To recalibrate for a different model or dataset:

```bash
# Entropy threshold
python scripts/calibrate_threshold.py \
  --signal entropy \
  --base_csv logs/base_model_eval.csv \
  --baseline_csv logs/baseline_eval.csv \
  --lora_path checkpoints_baseline/checkpoint-2000

# KL divergence threshold
python scripts/calibrate_threshold.py \
  --signal kl \
  --base_csv logs/base_model_eval.csv \
  --baseline_csv logs/baseline_eval.csv \
  --lora_path checkpoints_baseline/checkpoint-2000
```

### 4. Evaluate

Multi-run comparative evaluation across all models and datasets:

```bash
# MATH-500 (25 stochastic runs)
python scripts/evaluate.py \
  --dataset math500 \
  --n-runs 25 \
  --output-dir logs_comparative_eval \
  --baseline-checkpoint checkpoints_baseline/checkpoint-2000 \
  --sagrpo-checkpoint checkpoints_sa_grpo/checkpoint-2000 \
  --eagrpo-checkpoint checkpoints_ea_grpo/checkpoint-2000

# GSM8K
python scripts/evaluate.py \
  --dataset gsm8k \
  --n-runs 10 \
  --output-dir logs_comparative_eval_gsm8k

# AIME 2025
python scripts/evaluate.py \
  --dataset aime \
  --n-runs 20 \
  --output-dir logs_comparative_eval_aime \
  --max-tokens 4096

# Use a different model
python scripts/evaluate.py \
  --dataset math500 \
  --n-runs 10 \
  --output-dir logs_eval \
  --model Qwen/Qwen3-8B \
  --gpu-mem-util 0.8
```

---

## Architecture

The framework is modular along three axes — step creation, masking signal, and masking function — that can be mixed and matched independently:

```
┌─────────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│  Step Creation       │    │  Masking Signal   │    │  Masking Function   │
│  Strategy            │ →  │                   │ →  │                     │
│                      │    │  • KL Divergence   │    │  • Heaviside Step   │
│  • Prompted Steps    │    │  • Entropy         │    │  • (Decay, Soft...) │
│  • (Fine-tuned...)   │    │  • (Gradient...)   │    │                     │
└─────────────────────┘    └──────────────────┘    └─────────────────────┘
```

| Component | What it does | Current implementation |
|---|---|---|
| **Step Creation Strategy** | How steps are elicited from the model | Prompt-instructed `<step>` tags + reward shaping |
| **Masking Signal** | Per-step metric that determines masking | KL divergence (SA-GRPO), Shannon entropy (EA-GRPO) |
| **Masking Function** | Converts per-step scores → binary mask | Heaviside step function (mask before branching point) |

### Training Modes

| Mode | Command | Description |
|---|---|---|
| **Baseline** | `--mode baseline` | Standard GRPO, no masking (control) |
| **SA-GRPO** | `--mode kl` | KL divergence masking (threshold: 0.0009) |
| **EA-GRPO** | `--mode entropy` | Entropy masking (threshold: 0.075) |

### How Step Masking Works

1. **Step Detection**: The model is prompted to produce `<step>...</step>` tags. The boundary detector maps these (and standalone `$$...$$` LaTeX blocks) to token-level ranges.
2. **Signal Computation**: A `MaskingSignal` computes a per-step scalar (e.g., mean KL divergence or mean entropy for each step).
3. **Mask Construction**: A `MaskingFunction` (e.g., `HeavisideStepMask`) walks the steps, finds the first step where the signal exceeds the threshold (the "branching point"), and masks all tokens before it.
4. **Loss Masking**: The per-token loss is multiplied by the step mask, zeroing out gradient contributions from confident/redundant steps.

> **Current limitation**: The hard Heaviside mask is effectively *prefix masking* — it masks all tokens *before* the first branching step and keeps the entire remainder of the response unmasked. It does not yet do selective, per-step updates (see [Roadmap](#roadmap--todo)).

---

## Project Structure

```
Step_Based_Updates/
├── step_grpo/                      # Core framework
│   ├── __init__.py
│   ├── steps/                      # Step creation & detection
│   │   ├── strategy.py             # PromptedStepStrategy
│   │   └── boundary_detector.py    # <step> + $$ → token ranges
│   ├── masking/                    # Step-level masking
│   │   ├── signals.py              # MaskingSignal base + KL, Entropy signals
│   │   ├── functions.py            # MaskingFunction base + HeavisideStepMask
│   │   └── thresholds.py           # Default calibrated thresholds
│   ├── rewards/                    # Reward functions
│   │   └── math_reward.py          # Compound math reward (max 1.0)
│   ├── trainers/                   # GRPO trainer
│   │   ├── base_step_grpo.py       # StepGRPOTrainer (unified)
│   │   └── logging.py              # CSV + JSON training logger
│   └── evaluation/                 # Evaluation utilities
│       └── answer_matching.py      # Answer extraction & matching
├── configs/                        # Training configurations
│   ├── baseline.py                 # No masking
│   ├── kl_masking.py               # SA-GRPO
│   └── entropy_masking.py          # EA-GRPO
├── scripts/                        # Entry points
│   ├── train.py                    # Unified training
│   ├── evaluate.py                 # Multi-run comparative evaluation
│   ├── calibrate_threshold.py      # Threshold calibration
│   └── prepare_data.py             # Dataset download & formatting
├── data/                           # Datasets (JSON)
├── Dockerfile                      # Docker setup for reproducibility
└── README.md                       # This file
```

---

## Extending the Framework

The trainer uses a **generic signal dispatch** — it never checks `isinstance` for specific signal types. Each signal declares what inputs it needs via `required_inputs`, and the trainer maps them automatically. Adding a new signal or masking function requires **zero changes** to the trainer.

### Adding a New Masking Signal

```python
from step_grpo.masking.signals import MaskingSignal

class GradientNormSignal(MaskingSignal):
    name = "gradient_norm"
    required_inputs = ["grad_norms"]  # trainer maps this automatically

    def compute_per_step(self, step_boundaries, mask, **kwargs):
        grad_norms = kwargs["grad_norms"]  # (B, T)
        # ... compute mean grad norm per step ...
        return batch_scores
```

### Adding a New Masking Function

```python
from step_grpo.masking.functions import MaskingFunction

class ExponentialDecayMask(MaskingFunction):
    name = "exponential_decay"

    def __init__(self, threshold, decay_rate=0.5):
        self.threshold = threshold
        self.decay_rate = decay_rate

    def build_mask(self, per_step_scores, step_boundaries, advantages, seq_len, device):
        # ... smooth mask with exponential decay instead of hard cutoff ...
        return step_mask
```

### Adding a New Reward Function

```python
def code_reward_fn(prompts, completions, ground_truths, **kwargs):
    """Custom reward for code generation tasks."""
    rewards = []
    for comp, gt in zip(completions, ground_truths):
        # ... run test cases, check correctness ...
        rewards.append(score)
    return rewards
```

### Adding a New Step Strategy

```python
from step_grpo.steps.strategy import StepCreationStrategy

class CodeStepStrategy(StepCreationStrategy):
    name = "code_step"

    def get_prompt_template(self, question):
        return f"Solve step by step, wrapping each step in <step>...</step> tags.\n\n{question}"
```

---

## Threshold Analysis ✅

**Motivation (peer review)**: The step-masking idea was judged plausible, but the masking threshold was the weakest link — it is a single scalar (the mean per-step signal over "divergent" items) and was not backed by a distributional analysis. The default thresholds (KL = 0.0009, entropy = 0.075) were validated with the analysis below and updated where the data warranted it.

**Approach**:

1. **Per-step signal curves** — graph per-step KL divergence and Shannon entropy as a function of step index for the base Qwen3-4B, baseline GRPO, SA-GRPO, and EA-GRPO on MATH-500, split by outcome (stable-correct, stable-incorrect, divergent where correctness flips between models).
2. **Distribution analysis** — histogram / KDE of per-step KL and entropy values to locate where "confident/redundant" steps separate from "uncertain/divergent" steps, and pick a defensible threshold (knee point, quantile, or per-position cutoff).
3. **Robustness** — check how the threshold varies across subjects, difficulty levels, and step positions; decide between one global threshold and per-position / per-stratum thresholds.

**Deliverables**:
- Graphs → `notebooks/threshold_analysis.ipynb` (or a new `scripts/analyze_thresholds.py`)
- Updated `step_grpo/masking/thresholds.py` with the recommended threshold(s) and the evidence behind them

---

## Roadmap / TODO

- [x] **Threshold analysis** — graph KL and entropy across steps on MATH-500 (base vs. baseline GRPO, divergent vs. stable items) and pick a data-driven masking threshold. See [Threshold Analysis](#threshold-analysis) above.
- [ ] **Selective step-only updates** — move beyond prefix masking. After the threshold is settled, replace `HeavisideStepMask`'s "mask everything before the branching point" behavior with true per-step selective updates: only the targeted steps (where the signal crosses the calibrated threshold) receive gradient updates, while confident/redundant steps are masked regardless of their position in the response.
- [ ] **Soft masking functions** — exponential-decay / weighted masks instead of the hard Heaviside cutoff (base class already stubbed in `step_grpo/masking/functions.py`).
- [ ] **Additional signals** — e.g., a gradient-norm-based per-step signal via the existing `required_inputs` dispatch (no trainer changes needed).
- [ ] **Domain generalization** — step prompts, rewards, and boundary detection beyond math (code, science).

---

## Hyperparameters

All three models use identical training configs for fair comparison:

| Parameter | Value |
|---|---|
| Base model | Qwen/Qwen3-4B |
| LoRA rank (r) | 32 |
| LoRA alpha | 64 |
| LoRA dropout | 0.05 |
| LoRA targets | q, k, v, o, gate, up, down projections |
| Group size (k) | 4 |
| KL penalty (β) | 0.01 |
| Clip epsilon (ε) | 0.2 |
| Learning rate | 1e-5 |
| Max completion tokens | 1024 |
| Temperature | 0.7 |
| Top-p | 0.9 |
| Batch size × grad accum | 2 × 2 = 4 effective |
| Training data | 2,000 stratified math problems |
| Epochs | 1 |
| Gradient checkpointing | ✓ |
| Precision | bf16 |
| vLLM (generation) | Colocated mode, 50% GPU mem |
| **GPU** | **1× NVIDIA L40S (32 GB VRAM) — Cloud** |

### Masking Thresholds

| Signal | Default Threshold | Calibration |
|---|---|---|
| KL Divergence (SA-GRPO) | 0.0009 | Mean per-step KL on divergent items (base vs. baseline) |
| Entropy (EA-GRPO) | 0.075 | Mean per-step entropy on divergent items (base vs. baseline) |

---

## Requirements

| Dependency | Purpose |
|---|---|
| PyTorch ≥ 2.9 | Core ML framework |
| CUDA ≥ 13 | GPU compute |
| `trl[vllm]` | GRPO trainer + vLLM inference |
| `peft` | LoRA adapters |
| `transformers` | Model loading + tokenization |
| `datasets` | HuggingFace dataset loading |
| `pandas` | Data analysis |
| `sympy` | Symbolic math answer matching |
| `accelerate` | Distributed/mixed-precision training |

**Install all at once:**

```bash
uv pip install peft "trl[vllm]" pandas datasets sympy accelerate tqdm numpy
```

---

## Full Workflow (End-to-End)

```bash
# ── 1. Setup (Cloud NVIDIA L40S, CUDA 13, Python 3.12) ──
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc
uv pip install peft "trl[vllm]" pandas datasets sympy accelerate tqdm numpy

# ── 2. Prepare evaluation datasets ──
python scripts/prepare_data.py

# ── 3. Train all three models ──
python scripts/train.py --mode baseline
python scripts/train.py --mode kl
python scripts/train.py --mode entropy

# ── 4. Evaluate on MATH-500 ──
python scripts/evaluate.py \
  --dataset math500 --n-runs 25 \
  --output-dir logs_comparative_eval \
  --baseline-checkpoint checkpoints_baseline/checkpoint-2000 \
  --sagrpo-checkpoint checkpoints_sa_grpo/checkpoint-2000 \
  --eagrpo-checkpoint checkpoints_ea_grpo/checkpoint-2000

# ── 5. Evaluate on GSM8K ──
python scripts/evaluate.py \
  --dataset gsm8k --n-runs 10 \
  --output-dir logs_comparative_eval_gsm8k

# ── 6. Evaluate on AIME 2025 ──
python scripts/evaluate.py \
  --dataset aime --n-runs 20 \
  --output-dir logs_comparative_eval_aime --max-tokens 4096
```
