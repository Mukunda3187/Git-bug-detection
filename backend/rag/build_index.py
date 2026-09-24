
"""
Build a lightweight TF-IDF-only RAG knowledge-base index.

Generated files:
- metadata.json
- vectorizer.joblib
"""

import glob
import json
import os

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


DATASETS_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "datasets",
        "normalized",
    )
)

INDEX_DIR = os.path.join(
    os.path.dirname(__file__),
    "index",
)

METADATA_PATH = os.path.join(
    INDEX_DIR,
    "metadata.json",
)

TFIDF_VECTORIZER_PATH = os.path.join(
    INDEX_DIR,
    "vectorizer.joblib",
)


def _synthesize_description(record):
    bug_type = record.get("bug_type")
    error = record.get("error")
    language = record.get("language") or "code"

    if bug_type and error:
        return (
            f"A {str(bug_type).lower()} in "
            f"{language} code: {error}"
        )

    if bug_type:
        return (
            f"A {str(bug_type).lower()} found "
            f"in {language} code."
        )

    if error:
        return f"{language} code that raises: {error}"

    return f"A bug fixed in {language} code."


def load_all_records():
    """Load valid normalized JSONL records."""
    records = []

    if not os.path.isdir(DATASETS_DIR):
        print(
            f"[build_index] Dataset directory not found: "
            f"{DATASETS_DIR}"
        )
        return records

    dataset_paths = sorted(
        glob.glob(
            os.path.join(DATASETS_DIR, "*.jsonl")
        )
    )

    print(
        f"[build_index] Found {len(dataset_paths)} dataset files."
    )

    for path in dataset_paths:
        print(
            f"[build_index] Reading {os.path.basename(path)}"
        )

        try:
            with open(path, "r", encoding="utf-8") as f:
                for line_number, line in enumerate(f, 1):
                    line = line.strip()

                    if not line:
                        continue

                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        print(
                            f"[build_index] Invalid JSON in "
                            f"{os.path.basename(path)}:"
                            f"{line_number}: {exc}"
                        )
                        continue

                    if not isinstance(record, dict):
                        continue

                    if not record.get("buggy_code"):
                        continue

                    if not record.get("bug_description"):
                        record["bug_description"] = (
                            _synthesize_description(record)
                        )

                    record["_dataset_source"] = os.path.basename(path)
                    records.append(record)

        except Exception as exc:
            print(
                f"[build_index] Failed reading {path}: {exc}"
            )

    return records


def embedding_text(record):
    """Create the text representation used by TF-IDF."""

    parts = []

    if record.get("bug_description"):
        parts.append(str(record["bug_description"]))

    if record.get("error"):
        parts.append(f"Error: {record['error']}")

    if record.get("bug_type"):
        parts.append(f"Type: {record['bug_type']}")

    if record.get("language"):
        parts.append(f"Language: {record['language']}")

    if record.get("buggy_code"):
        parts.append(f"Code: {record['buggy_code']}")

    if record.get("solution"):
        parts.append(f"Solution: {record['solution']}")

    return "\n".join(filter(None, parts))


def build_tfidf_index(texts):
    """Build and save the TF-IDF vectorizer."""

    print("[build_index] Building TF-IDF index...")

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
    )

    vectorizer.fit(texts)

    joblib.dump(
        vectorizer,
        TFIDF_VECTORIZER_PATH,
    )

    print(
        f"[build_index] TF-IDF vectorizer saved: "
        f"{TFIDF_VECTORIZER_PATH}"
    )


def save_metadata(records):
    """Save the original normalized records."""

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(
            records,
            f,
            ensure_ascii=False,
        )

    print(
        f"[build_index] Metadata saved: {METADATA_PATH}"
    )


def build():
    """Build the TF-IDF-only RAG index."""

    records = load_all_records()

    if not records:
        print("[build_index] No valid records found.")
        return

    os.makedirs(INDEX_DIR, exist_ok=True)

    texts = [
        embedding_text(record)
        for record in records
    ]

    try:
        build_tfidf_index(texts)
        save_metadata(records)

        print(
            f"[build_index] TF-IDF index ready: "
            f"{len(records)} records."
        )

    except Exception as exc:
        print(
            f"[build_index] Failed to build TF-IDF index: {exc}"
        )
        raise


if __name__ == "__main__":
    build()
