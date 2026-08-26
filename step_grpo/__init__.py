"""Step-GRPO: A framework for step-level RLVR.

Uses the reasoning step as the fundamental unit of reinforcement,
rather than individual tokens. Masking strategies focus RL signal
on steps where the model is uncertain or divergent, skipping
confident/redundant reasoning.
"""

__version__ = "0.1.0"
