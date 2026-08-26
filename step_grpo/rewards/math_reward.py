"""Math reward function for GRPO training.

Compound reward that incentivizes:
  - Correct structural formatting (<think>, <answer>, <step> tags)
  - Correct mathematical answers (verified via SymPy)

Note: This reward function is specific to mathematical reasoning tasks.
For other domains (code, science, etc.), implement a custom reward
function with the same signature: (prompts, completions, ground_truths) -> List[float].
"""

import re
from typing import List

from step_grpo.evaluation.answer_matching import extract_answer, answers_match


_STEP_TAG_RE = re.compile(r"<step>", re.DOTALL)


def math_reward_fn(
    prompts: List[str],
    completions: List[str],
    ground_truths: List[str],
    **kwargs,
) -> List[float]:
    """Compute compound math reward for each completion.

    Reward breakdown:
      +0.05  Correct <think>...</think> placement
      +0.05  Correct <answer>...</answer> placement
      +0.05  At least 2 <step> tags present
      +0.85  Correct mathematical answer

    Total max: 1.0
    """
    rewards = []
    for comp, gt in zip(completions, ground_truths):
        score = 0.0
        if "<think>" in comp and "</think>" in comp:
            score += 0.05
        if "<answer>" in comp and "</answer>" in comp:
            score += 0.05
        if len(_STEP_TAG_RE.findall(comp)) >= 2:
            score += 0.05
        extracted = extract_answer(comp)
        if extracted and answers_match(extracted, gt):
            score += 0.85
        rewards.append(round(score, 4))
    return rewards
