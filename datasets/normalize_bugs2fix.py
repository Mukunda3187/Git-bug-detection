"""
Downloads Bugs2Fix (CodeXGLUE "code-refinement" task) from Hugging Face
and converts it into our normalized schema.

Run this on a machine with internet access to huggingface.co:
    pip install datasets
    python normalize_bugs2fix.py

Output: datasets/normalized/bugs2fix.jsonl
"""
import json
import os
import sys

sys.path.append(os.path.dirname(__file__))
from schema import NormalizedBug

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "normalized", "bugs2fix.jsonl")


def main():
    from datasets import load_dataset

    # "medium" config = Bugs2Fix medium-length functions. Use "small" if you
    # want a quicker download for a mini-project demo.
    ds = load_dataset("google/code_x_glue_cc_code_refinement", "medium", split="train")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    count = 0
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for i, row in enumerate(ds):
            buggy = row.get("buggy", "").strip()
            fixed = row.get("fixed", "").strip()
            if not buggy or not fixed:
                continue
            entry = NormalizedBug(
                id=f"bugs2fix_{i}",
                dataset_source="Bugs2Fix",
                language="java",
                buggy_code=buggy,
                fixed_code=fixed,
                error=None,                # CodeXGLUE does not include a runtime error message
                bug_type=None,             # not labeled in the raw dataset
                bug_description=None,      # not provided - left empty, not invented
                solution=None,
                repository=None,
                file=None,
                metadata={"source_split": "train"},
            )
            out.write(json.dumps(entry.to_dict()) + "\n")
            count += 1

    print(f"Wrote {count} normalized entries to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
