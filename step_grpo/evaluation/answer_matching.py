"""Answer extraction and matching utilities for math verification."""

import re
from typing import Optional


_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_STEP_TAG_RE = re.compile(r"<step>", re.DOTALL)
_LAST_NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")


def strip_latex_wrappers(s: str) -> str:
    """Remove common LaTeX formatting wrappers from a string."""
    s = s.strip()
    s = re.sub(r"^\$+|\$+$", "", s)
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = s.replace("\\left", "").replace("\\right", "")
    return s.rstrip(".").strip()


def extract_answer(text: str) -> str:
    """Extract the final answer from a model response.

    Priority: <answer> tags > \\boxed{} > last non-empty line.
    """
    m = _ANSWER_TAG_RE.findall(text)
    if m:
        return strip_latex_wrappers(m[-1])
    boxed = _BOXED_RE.findall(text)
    if boxed:
        return strip_latex_wrappers(boxed[-1])
    lines = [strip_latex_wrappers(l) for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else ""


def normalize(s: str) -> str:
    """Normalize a math answer string for comparison."""
    s = strip_latex_wrappers(s)
    s = s.lower().replace(",", "").replace(" ", "")
    frac_match = re.match(r"^-?\\frac\{(\d+)\}\{(\d+)\}$", s)
    if frac_match:
        sign = -1 if s.startswith("-") else 1
        nums = re.findall(r"\d+", s)
        if len(nums) == 2:
            return str(sign * int(nums[0]) / int(nums[1]))
    try:
        return str(float(s))
    except ValueError:
        return s


def answers_match(pred: str, gt: str) -> bool:
    """Check if a predicted answer matches the ground truth.

    Uses multiple strategies: exact match, normalized match,
    float tolerance, and SymPy symbolic comparison.
    """
    if strip_latex_wrappers(pred) == strip_latex_wrappers(gt):
        return True
    if normalize(pred) == normalize(gt):
        return True

    try:
        if abs(float(normalize(pred)) - float(normalize(gt))) < 1e-2:
            return True
    except ValueError:
        pass

    try:
        from sympy.parsing.latex import parse_latex
        from sympy import simplify, N
        diff = simplify(parse_latex(strip_latex_wrappers(pred)) - parse_latex(strip_latex_wrappers(gt)))
        if diff == 0:
            return True
        if abs(float(N(diff))) < 1e-2:
            return True
    except Exception:
        pass

    try:
        from sympy import simplify, sympify, N
        diff = simplify(sympify(strip_latex_wrappers(pred)) - sympify(strip_latex_wrappers(gt)))
        if diff == 0:
            return True
        if abs(float(N(diff))) < 1e-2:
            return True
    except Exception:
        return False
    return False


def answers_match_numeric(pred: str, gt: str) -> bool:
    """Strict numeric matching for GSM8K/AIME (answers are always numbers)."""
    pred_clean = strip_latex_wrappers(pred).replace(",", "").strip()
    gt_clean = gt.replace(",", "").strip()

    if pred_clean == gt_clean:
        return True

    try:
        if abs(float(pred_clean) - float(gt_clean)) < 1e-2:
            return True
    except ValueError:
        pass

    nums = _LAST_NUM_RE.findall(pred_clean)
    if nums:
        try:
            last_num = nums[-1].replace(",", "")
            if abs(float(last_num) - float(gt_clean)) < 1e-2:
                return True
        except ValueError:
            pass

    return False
