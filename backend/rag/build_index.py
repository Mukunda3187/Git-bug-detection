"""
Build the RAG knowledge-base index.

This module creates:
1. Semantic embeddings using Sentence Transformers
2. A FAISS vector index for fast similarity search
3. Metadata containing the original bug records
4. A TF-IDF fallback index

Embedding model:
    all-MiniLM-L6-v2

Generated files:
    rag/index/embeddings.npy
    rag/index/faiss.index
    rag/index/metadata.json
    rag/index/vectorizer.joblib
"""

import glob
import json
import os

import joblib
import numpy as np
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

EMBEDDINGS_PATH = os.path.join(
    INDEX_DIR,
    "embeddings.npy",
)

FAISS_INDEX_PATH = os.path.join(
    INDEX_DIR,
    "faiss.index",
)

METADATA_PATH = os.path.join(
    INDEX_DIR,
    "metadata.json",
)

TFIDF_VECTORIZER_PATH = os.path.join(
    INDEX_DIR,
    "vectorizer.joblib",
)

MODEL_NAME = "all-MiniLM-L6-v2"


def _synthesize_description(record):
    """
    Create a short description when a dataset record does not
    contain bug_description.
    """

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
        return (
            f"{language} code that raises: {error}"
        )

    return (
        f"A bug fixed in {language} code."
    )


def load_all_records():
    """
    Load all valid normalized JSONL records.
    """

    records = []

    if not os.path.isdir(DATASETS_DIR):
        print(
            f"[build_index] Dataset directory not found: "
            f"{DATASETS_DIR}"
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
        f"[build_index] Found {len(dataset_paths)} dataset files."
    )

    for path in dataset_paths:

        print(
            f"[build_index] Reading {os.path.basename(path)}"
        )

        try:

            with open(
                path,
                "r",
                encoding="utf-8",
            ) as f:

                for line_number, line in enumerate(
                    f,
                    1,
                ):

                    line = line.strip()

                    if not line:
                        continue

                    try:

                        record = json.loads(
                            line
                        )

                    except json.JSONDecodeError as e:

                        print(
                            f"[build_index] Invalid JSON in "
                            f"{os.path.basename(path)}:"
                            f"{line_number}: {e}"
                        )

                        continue

                    if not isinstance(
                        record,
                        dict,
                    ):
                        continue

                    if not record.get(
                        "buggy_code"
                    ):
                        continue

                    if not record.get(
                        "bug_description"
                    ):

                        record[
                            "bug_description"
                        ] = _synthesize_description(
                            record
                        )

                    record[
                        "_dataset_source"
                    ] = os.path.basename(
                        path
                    )

                    records.append(
                        record
                    )

        except Exception as e:

            print(
                f"[build_index] Failed reading "
                f"{path}: {e}"
            )

    return records


def embedding_text(record):
    """
    Create the text representation used for semantic embeddings.

    This format must remain consistent with retriever.py.
    """

    parts = []

    if record.get(
        "bug_description"
    ):

        parts.append(
            str(
                record[
                    "bug_description"
                ]
            )
        )

    if record.get(
        "error"
    ):

        parts.append(
            f"Error: {record['error']}"
        )

    if record.get(
        "bug_type"
    ):

        parts.append(
            f"Type: {record['bug_type']}"
        )

    if record.get(
        "language"
    ):

        parts.append(
            f"Language: {record['language']}"
        )

    if record.get(
        "buggy_code"
    ):

        parts.append(
            f"Code: {record['buggy_code']}"
        )

    if record.get(
        "solution"
    ):

        parts.append(
            f"Solution: {record['solution']}"
        )

    return "\n".join(
        filter(
            None,
            parts,
        )
    )


def build_semantic_index(
    records,
    texts,
):
    """
    Generate Sentence Transformer embeddings and build FAISS index.
    """

    print(
        f"[build_index] Loading embedding model: "
        f"{MODEL_NAME}"
    )

    from sentence_transformers import (
        SentenceTransformer,
    )

    import faiss

    model = SentenceTransformer(
        MODEL_NAME
    )

    print(
        "[build_index] Generating semantic embeddings..."
    )

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32",
    )

    if embeddings.ndim != 2:
        raise ValueError(
            "Semantic embeddings must be a 2D array."
        )

    print(
        f"[build_index] Embedding shape: "
        f"{embeddings.shape}"
    )

    # Save raw embeddings as well.
    np.save(
        EMBEDDINGS_PATH,
        embeddings,
    )

    # Because embeddings are normalized, inner product is equivalent
    # to cosine similarity.
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    faiss.write_index(
        index,
        FAISS_INDEX_PATH,
    )

    print(
        f"[build_index] FAISS index saved: "
        f"{FAISS_INDEX_PATH}"
    )

    print(
        f"[build_index] FAISS records: "
        f"{index.ntotal}"
    )


def build_tfidf_index(
    texts,
):
    """
    Build the lightweight TF-IDF fallback.
    """

    print(
        "[build_index] Building TF-IDF fallback..."
    )

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
    )

    vectorizer.fit(
        texts
    )

    joblib.dump(
        vectorizer,
        TFIDF_VECTORIZER_PATH,
    )

    print(
        f"[build_index] TF-IDF vectorizer saved: "
        f"{TFIDF_VECTORIZER_PATH}"
    )


def save_metadata(
    records,
):
    """
    Save the original normalized records.
    """

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            records,
            f,
            ensure_ascii=False,
        )

    print(
        f"[build_index] Metadata saved: "
        f"{METADATA_PATH}"
    )


def build():
    """
    Build the complete RAG index.
    """

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

    print(
        f"[build_index] Loaded "
        f"{len(records)} valid records."
    )

    texts = [
        embedding_text(
            record
        )
        for record in records
    ]

    # ---------------------------------------------------------
    # Semantic embeddings + FAISS
    # ---------------------------------------------------------

    semantic_success = False

    try:

        build_semantic_index(
            records,
            texts,
        )

        semantic_success = True

    except Exception as e:

        print(
            "[build_index] WARNING: Semantic/FAISS "
            f"index creation failed: {e}"
        )

        print(
            "[build_index] Continuing with TF-IDF fallback."
        )

    # ---------------------------------------------------------
    # TF-IDF fallback
    # ---------------------------------------------------------

    try:

        build_tfidf_index(
            texts
        )

    except Exception as e:

        print(
            f"[build_index] WARNING: TF-IDF build failed: {e}"
        )

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    try:

        save_metadata(
            records
        )

    except Exception as e:

        print(
            f"[build_index] Failed to save metadata: {e}"
        )

        return

    # ---------------------------------------------------------
    # Final status
    # ---------------------------------------------------------

    print()
    print(
        "=============================================="
    )

    print(
        "RAG INDEX BUILD COMPLETE"
    )

    print(
        f"Records: {len(records)}"
    )

    print(
        f"Semantic embeddings: "
        f"{'READY' if semantic_success else 'FAILED'}"
    )

    print(
        f"FAISS index: "
        f"{'READY' if os.path.exists(FAISS_INDEX_PATH) else 'FAILED'}"
    )

    print(
        f"TF-IDF fallback: "
        f"{'READY' if os.path.exists(TFIDF_VECTORIZER_PATH) else 'FAILED'}"
    )

    print(
        "=============================================="
    )


if __name__ == "__main__":
    build()
