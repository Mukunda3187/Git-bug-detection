"""
Build the RAG knowledge-base index.

Current implementation:
- Loads all normalized JSONL bug records.
- Creates a TF-IDF representation.
- Saves the TF-IDF vectorizer.
- Saves metadata used by retriever.py.

This version is designed to work with the updated retriever.py.
"""

import glob
import json
import os
from typing import Dict, List

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
    )
)

DATASETS_DIR = os.path.join(
    BASE_DIR,
    "datasets",
    "normalized",
)

INDEX_DIR = os.path.join(
    os.path.dirname(__file__),
    "index",
)

METADATA_PATH = os.path.join(
    INDEX_DIR,
    "metadata.json",
)

VECTORIZER_PATH = os.path.join(
    INDEX_DIR,
    "vectorizer.joblib",
)


# ============================================================
# DESCRIPTION HELPERS
# ============================================================

def _synthesize_description(record: Dict) -> str:
    """
    Create a bug description when the dataset record
    does not already contain one.
    """

    bug_type = record.get("bug_type")
    error = record.get("error")
    language = record.get("language") or "code"

    if bug_type and error:
        return (
            f"A {str(bug_type).lower()} bug in "
            f"{language} code: {error}"
        )

    if bug_type:
        return (
            f"A {str(bug_type).lower()} bug found "
            f"in {language} code."
        )

    if error:
        return (
            f"{language} code that produces "
            f"the following error: {error}"
        )

    return f"A bug fixed in {language} code."


# ============================================================
# DATASET LOADING
# ============================================================

def load_all_records() -> List[Dict]:
    """
    Load all valid records from every JSONL file
    inside datasets/normalized.
    """

    records: List[Dict] = []

    if not os.path.isdir(DATASETS_DIR):
        print(
            "[build_index] Dataset directory not found:"
        )
        print(
            f"               {DATASETS_DIR}"
        )
        return records

    dataset_paths = sorted(
        glob.glob(
            os.path.join(
                DATASETS_DIR,
                "*.jsonl",
            )
        )
    )

    print(
        f"[build_index] Found "
        f"{len(dataset_paths)} dataset file(s)."
    )

    for dataset_path in dataset_paths:

        filename = os.path.basename(
            dataset_path
        )

        print(
            f"[build_index] Reading: {filename}"
        )

        try:

            with open(
                dataset_path,
                "r",
                encoding="utf-8",
            ) as file:

                for line_number, line in enumerate(
                    file,
                    start=1,
                ):

                    line = line.strip()

                    if not line:
                        continue

                    try:
                        record = json.loads(line)

                    except json.JSONDecodeError as exc:
                        print(
                            f"[build_index] Invalid JSON "
                            f"in {filename}:{line_number}"
                        )
                        print(
                            f"               {exc}"
                        )
                        continue

                    if not isinstance(
                        record,
                        dict,
                    ):
                        continue

                    # A usable historical example should
                    # contain buggy code.
                    if not record.get(
                        "buggy_code"
                    ):
                        continue

                    # Create missing description.
                    if not record.get(
                        "bug_description"
                    ):
                        record[
                            "bug_description"
                        ] = _synthesize_description(
                            record
                        )

                    # Keep track of the dataset.
                    record[
                        "_dataset_source"
                    ] = filename

                    records.append(record)

        except Exception as exc:

            print(
                f"[build_index] Failed reading "
                f"{filename}: {exc}"
            )

    print(
        f"[build_index] Loaded "
        f"{len(records)} valid record(s)."
    )

    return records


# ============================================================
# TEXT CREATION
# ============================================================

def embedding_text(record: Dict) -> str:
    """
    Convert a historical bug record into searchable text.

    The retriever uses this same structure when searching
    for similar historical bugs.
    """

    parts: List[str] = []

    bug_description = record.get(
        "bug_description"
    )

    if bug_description:
        parts.append(
            str(bug_description)
        )

    error = record.get("error")

    if error:
        parts.append(
            f"Error: {error}"
        )

    bug_type = record.get(
        "bug_type"
    )

    if bug_type:
        parts.append(
            f"Bug Type: {bug_type}"
        )

    language = record.get(
        "language"
    )

    if language:
        parts.append(
            f"Language: {language}"
        )

    buggy_code = record.get(
        "buggy_code"
    )

    if buggy_code:
        parts.append(
            f"Buggy Code: {buggy_code}"
        )

    solution = record.get(
        "solution"
    )

    if solution:
        parts.append(
            f"Solution: {solution}"
        )

    fixed_code = (
        record.get("fixed_code")
        or record.get("corrected_code")
        or record.get("replacement_code")
    )

    if fixed_code:
        parts.append(
            f"Fixed Code: {fixed_code}"
        )

    return "\n".join(
        str(part)
        for part in parts
        if part
    )


# ============================================================
# TF-IDF INDEX
# ============================================================

def build_tfidf_index(
    texts: List[str],
) -> None:
    """
    Build and save the TF-IDF vectorizer.
    """

    if not texts:
        raise ValueError(
            "No searchable text was generated."
        )

    print(
        "[build_index] Building TF-IDF vectorizer..."
    )

    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words="english",
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    vectorizer.fit(texts)

    joblib.dump(
        vectorizer,
        VECTORIZER_PATH,
    )

    print(
        "[build_index] Vectorizer saved:"
    )
    print(
        f"               {VECTORIZER_PATH}"
    )


# ============================================================
# METADATA
# ============================================================

def save_metadata(
    records: List[Dict],
) -> None:
    """
    Save historical bug records used by the retriever.
    """

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            records,
            file,
            ensure_ascii=False,
        )

    print(
        "[build_index] Metadata saved:"
    )
    print(
        f"               {METADATA_PATH}"
    )


# ============================================================
# BUILD
# ============================================================

def build() -> None:
    """
    Build the complete TF-IDF RAG index.
    """

    print(
        "\n=========================================="
    )
    print(
        "        BUILDING RAG KNOWLEDGE BASE"
    )
    print(
        "=========================================="
    )

    records = load_all_records()

    if not records:

        print(
            "[build_index] No valid records found."
        )

        return

    os.makedirs(
        INDEX_DIR,
        exist_ok=True,
    )

    texts = [
        embedding_text(record)
        for record in records
    ]

    # Remove completely empty records.
    valid_pairs = [
        (record, text)
        for record, text in zip(
            records,
            texts,
        )
        if text.strip()
    ]

    if not valid_pairs:

        print(
            "[build_index] No searchable records found."
        )

        return

    records = [
        pair[0]
        for pair in valid_pairs
    ]

    texts = [
        pair[1]
        for pair in valid_pairs
    ]

    print(
        f"[build_index] Searchable records: "
        f"{len(records)}"
    )

    build_tfidf_index(
        texts
    )

    save_metadata(
        records
    )

    print(
        "\n[build_index] RAG index ready."
    )

    print(
        f"[build_index] Records: "
        f"{len(records)}"
    )

    print(
        f"[build_index] Index directory: "
        f"{INDEX_DIR}"
    )

    print(
        "==========================================\n"
    )


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":
    build()
