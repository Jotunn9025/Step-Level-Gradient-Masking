"""Step creation strategies for reasoning models.

A StepCreationStrategy defines:
  1. How to prompt the model to produce structured steps
  2. What reward shaping to apply for step compliance
  3. How to detect steps in the output (delegates to boundary_detector)

Currently, the only strategy is PromptedStepStrategy, which uses
system prompt instructions + structural reward shaping. Future strategies
could include fine-tuned step segmentation, post-hoc step splitting, etc.

Note: The included PromptedStepStrategy is math-specific. For other
domains, subclass StepCreationStrategy and provide domain-appropriate
prompt templates and step detection.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StepCreationStrategy:
    """Base class for step creation strategies.

    Defines the contract for how structured reasoning steps are
    elicited from a language model during GRPO rollouts.
    """
    name: str = "base"

    def get_prompt_template(self, question: str) -> str:
        """Return the full prompt including step-creation instructions."""
        raise NotImplementedError

    def get_step_reward_weight(self) -> float:
        """Return the reward weight allocated to step compliance."""
        return 0.0


@dataclass
class PromptedStepStrategy(StepCreationStrategy):
    """Elicit steps via prompt instructions + reward shaping.

    The model is instructed to:
      - Wrap reasoning in <think>...</think> tags
      - Break reasoning into <step>...</step> blocks
      - Use $$...$$ for LaTeX equations
      - Provide final answer in <answer>...</answer> tags

    Reward shaping allocates:
      +0.05 for <think> compliance
      +0.05 for <answer> compliance
      +0.05 for producing ≥2 <step> tags
    """
    name: str = "prompted_step"

    SYSTEM_TEMPLATE = (
        "Solve the following math problem step by step.\n"
        "Show your reasoning inside <think> tags, with each step in <step> tags.\n"
        "Then give your final answer inside <answer> tags.\n\n"
        "Format:\n"
        "<think>\n"
        "<step>First step</step>\n"
        "<step>Second step</step>\n"
        "Each and every bit of text including latex expressions should have a "
        "<step></step> surrounding them.\n"
        "</think>\n"
        "<answer>your answer</answer>\n\n"
    )

    def get_prompt_template(self, question: str) -> str:
        """Build the full prompt with step-creation instructions."""
        return f"{self.SYSTEM_TEMPLATE}Problem: {question}\n"

    def get_step_reward_weight(self) -> float:
        """Step compliance accounts for 0.15 of the total 1.0 reward."""
        return 0.15
