"""Step boundary detection in model completions.

Maps structured reasoning markers (<step> tags and $$ LaTeX blocks)
to token-level (start, end) ranges for downstream masking.
"""

from __future__ import annotations

import re
from typing import List, Tuple

import torch


# Patterns for detecting reasoning step boundaries
STEP_TAG_RE = re.compile(r"<step>(.*?)</step>", re.DOTALL)
LATEX_BLOCK_RE = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)


def find_step_boundaries(
    completion_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    tokenizer,
) -> List[List[Tuple[int, int]]]:
    """Find step boundaries in a batch of completions.

    Detects both <step>...</step> tags and standalone $$...$$ LaTeX
    display blocks (that are not already inside a <step> region).

    Args:
        completion_ids: (B, T) token IDs for completions.
        completion_mask: (B, T) attention mask for completions.
        tokenizer: HuggingFace tokenizer for decoding.

    Returns:
        For each sample in the batch, a list of (start_tok, end_tok) ranges.
    """
    batch_size = completion_ids.size(0)
    all_boundaries = []

    for b in range(batch_size):
        active_len = int(completion_mask[b].sum().item())
        if active_len == 0:
            all_boundaries.append([])
            continue

        active_ids = completion_ids[b, :active_len]
        text = tokenizer.decode(active_ids, skip_special_tokens=True)

        # Find <step>...</step> character spans
        step_spans = [(m.start(), m.end()) for m in STEP_TAG_RE.finditer(text)]

        # Find $$...$$ spans NOT overlapping with <step> regions
        for m in LATEX_BLOCK_RE.finditer(text):
            inside = any(s <= m.start() and m.end() <= e for s, e in step_spans)
            if not inside:
                step_spans.append((m.start(), m.end()))
        step_spans.sort(key=lambda x: x[0])

        if not step_spans:
            all_boundaries.append([])
            continue

        # Map character offsets → token offsets
        token_ranges = _char_spans_to_token_ranges(
            text, step_spans, active_len, tokenizer,
        )
        all_boundaries.append(token_ranges)

    return all_boundaries


def _char_spans_to_token_ranges(
    text: str,
    char_spans: List[Tuple[int, int]],
    active_len: int,
    tokenizer,
) -> List[Tuple[int, int]]:
    """Convert character-level spans to token-level ranges.

    Uses offset_mapping for precision, with a fallback to substring
    re-encoding if offset_mapping is unavailable.
    """
    token_ranges = []
    try:
        # Fast path: use offset_mapping for precise char→token mapping
        encoding = tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True,
        )
        offsets = encoding["offset_mapping"]
        if len(offsets) != active_len:
            raise ValueError("Token count mismatch after re-tokenization")

        for char_start, char_end in char_spans:
            tok_start = None
            tok_end = None
            for ti, (cs, ce) in enumerate(offsets):
                if ce > char_start and tok_start is None:
                    tok_start = ti
                if cs < char_end:
                    tok_end = ti + 1
            if tok_start is not None and tok_end is not None:
                tok_start = min(tok_start, active_len)
                tok_end = min(tok_end, active_len)
                if tok_end > tok_start:
                    token_ranges.append((tok_start, tok_end))
    except Exception:
        # Fallback: re-encode substrings (less precise due to BPE boundaries)
        for char_start, char_end in char_spans:
            prefix_text = text[:char_start]
            tok_start = len(tokenizer.encode(prefix_text, add_special_tokens=False))
            full_text = text[:char_end]
            tok_end = len(tokenizer.encode(full_text, add_special_tokens=False))
            tok_start = min(tok_start, active_len)
            tok_end = min(tok_end, active_len)
            if tok_end > tok_start:
                token_ranges.append((tok_start, tok_end))

    return token_ranges
