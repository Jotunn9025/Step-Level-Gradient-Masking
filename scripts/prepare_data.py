"""Dataset preparation — download and format evaluation datasets.

Produces:
  data/eval_gsm8k.json         — GSM8K test (1,319 problems)
  data/eval_aime.json          — AIME 2025 (30 problems)
  data/eval_math500_with_types.json — MATH-500 with subject + difficulty

Usage:
  python scripts/prepare_data.py
"""

import json
from pathlib import Path
from collections import Counter
from datasets import load_dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def prepare_gsm8k():
    """Download and format GSM8K test set."""
    print("=" * 60)
    print("  Downloading GSM8K test set...")
    print("=" * 60)

    gsm8k = load_dataset("openai/gsm8k", "main", split="test")
    print(f"  Loaded {len(gsm8k)} problems")

    gsm8k_eval = []
    for row in gsm8k:
        raw_answer = row["answer"]
        if "####" in raw_answer:
            gt = raw_answer.split("####")[-1].strip().replace(",", "")
        else:
            gt = raw_answer.strip().split("\n")[-1].strip()
        gsm8k_eval.append({"question": row["question"], "ground_truth": gt})

    out_path = DATA_DIR / "eval_gsm8k.json"
    with open(out_path, "w") as f:
        json.dump(gsm8k_eval, f, indent=2)
    print(f"  Saved {len(gsm8k_eval)} problems → {out_path}")


def prepare_aime():
    """Download and format AIME 2025."""
    print("\n" + "=" * 60)
    print("  Downloading AIME 2025...")
    print("=" * 60)

    aime = load_dataset("MathArena/aime_2025", split="train")
    print(f"  Loaded {len(aime)} problems")

    aime_eval = []
    for row in aime:
        aime_eval.append({
            "question": row["problem"],
            "ground_truth": str(row["answer"]),
        })

    out_path = DATA_DIR / "eval_aime.json"
    with open(out_path, "w") as f:
        json.dump(aime_eval, f, indent=2)
    print(f"  Saved {len(aime_eval)} problems → {out_path}")


def prepare_math500_enriched():
    """Enrich MATH-500 with subject and difficulty metadata."""
    print("\n" + "=" * 60)
    print("  Enriching MATH-500 with metadata...")
    print("=" * 60)

    eval_path = DATA_DIR / "eval.json"
    if not eval_path.exists():
        print(f"  ERROR: {eval_path} not found. Skipping enrichment.")
        return

    with open(eval_path) as f:
        math500 = json.load(f)
    print(f"  MATH-500 eval set: {len(math500)} problems")

    # Try loading MATH test set for metadata
    math_full = None
    for dataset_name in [
        "hendrycks/competition_math",
        "lighteval/MATH",
        "EleutherAI/hendrycks_math",
    ]:
        try:
            print(f"  Trying {dataset_name}...")
            math_full = load_dataset(dataset_name, split="test", trust_remote_code=True)
            print(f"  ✓ Loaded {len(math_full)} problems from {dataset_name}")
            break
        except Exception as e:
            print(f"  ✗ {e}")

    if math_full is None:
        print("  WARNING: Could not load MATH test set for metadata.")
        return

    cols = math_full.column_names
    q_col = "problem" if "problem" in cols else "question"
    type_col = "type" if "type" in cols else "subject" if "subject" in cols else None
    level_col = "level" if "level" in cols else "difficulty" if "difficulty" in cols else None

    metadata_lookup = {}
    for row in math_full:
        q_text = row[q_col].strip()
        entry = {}
        if type_col and type_col in row:
            entry["subject"] = row[type_col]
        if level_col and level_col in row:
            entry["difficulty"] = row[level_col]
        metadata_lookup[q_text] = entry

    matched = 0
    enriched = []
    for item in math500:
        q = item["question"].strip()
        meta = metadata_lookup.get(q, {})
        enriched.append({
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "subject": meta.get("subject", "Unknown"),
            "difficulty": meta.get("difficulty", "Unknown"),
        })
        if meta:
            matched += 1

    print(f"  Matched {matched}/{len(math500)} questions ({100*matched/len(math500):.1f}%)")

    subjects = Counter(e["subject"] for e in enriched)
    print(f"\n  Subject distribution:")
    for s, c in sorted(subjects.items()):
        print(f"    {s}: {c}")

    out_path = DATA_DIR / "eval_math500_with_types.json"
    with open(out_path, "w") as f:
        json.dump(enriched, f, indent=2)
    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    prepare_gsm8k()
    prepare_aime()
    prepare_math500_enriched()
    print("\n" + "=" * 60)
    print("  Dataset preparation complete!")
    print("=" * 60)
