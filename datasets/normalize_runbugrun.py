"""
Downloads RunBugRun from Hugging Face (ASSERT-KTH/RunBugRun) and converts
it into our normalized schema. RunBugRun is large (450k+ entries across 8
languages), so by default this script only keeps Python and Java entries
and caps the count - fine for a mini project RAG knowledge base.

Run on a machine with internet access:
    pip install datasets
    python normalize_runbugrun.py

Output: datasets/normalized/runbugrun.jsonl
"""
import json
import os
import sys

sys.path.append(os.path.dirname(__file__))
from schema import NormalizedBug

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "normalized", "runbugrun.jsonl")

KEEP_LANGUAGES = {"python", "java"}
MAX_ENTRIES_PER_LANGUAGE = 2000   # keep the knowledge base a manageable size for a mini project


def main():
    from datasets import load_dataset

    ds = load_dataset("ASSERT-KTH/RunBugRun", split="train", streaming=True)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    counts = {lang: 0 for lang in KEEP_LANGUAGES}
    written = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for i, row in enumerate(ds):
            lang = (row.get("language") or "").lower()
            if lang not in KEEP_LANGUAGES:
                continue
            if counts[lang] >= MAX_ENTRIES_PER_LANGUAGE:
                if all(c >= MAX_ENTRIES_PER_LANGUAGE for c in counts.values()):
                    break
                continue

            buggy = (row.get("buggy_code") or row.get("source_code") or "").strip()
            fixed = (row.get("fixed_code") or row.get("target_code") or "").strip()
            error_msg = row.get("error_message") or row.get("stderr") or None
            bug_type = row.get("error_class") or row.get("exception_type") or None

            if not buggy or not fixed:
                continue

            entry = NormalizedBug(
                id=f"runbugrun_{i}",
                dataset_source="RunBugRun",
                language=lang,
                buggy_code=buggy,
                fixed_code=fixed,
                error=error_msg,
                bug_type=bug_type,
                bug_description=None,
                solution=None,
                repository=None,
                file=None,
                metadata={"competitive_programming": True},
            )
            out.write(json.dumps(entry.to_dict()) + "\n")
            counts[lang] += 1
            written += 1

    print(f"Wrote {written} normalized entries to {OUTPUT_PATH}")
    print(f"Per-language counts: {counts}")


if __name__ == "__main__":
    main()
