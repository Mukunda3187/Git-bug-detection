"""
RAG retrieval layer.

Current index:
    TF-IDF + cosine similarity

The retriever is kept lightweight so repository scanning does not
consume unnecessary RAM.

The RAG layer retrieves historical bug examples and sends them
to the Gemini analysis layer as supporting evidence.
"""

import json
import os
import threading

INDEX_DIR = os.path.join(
    os.path.dirname(__file__),
    "index",
)

TFIDF_VECTORIZER_PATH = os.path.join(
    INDEX_DIR,
    "vectorizer.joblib",
)

METADATA_PATH = os.path.join(
    INDEX_DIR,
    "metadata.json",
)


# ============================================================
# IN-MEMORY INDEX
# ============================================================

_metadata = None
_vectorizer = None
_document_matrix = None

_load_lock = threading.Lock()


# ============================================================
# TEXT PREPARATION
# ============================================================

def _embedding_text(record: dict) -> str:
    """
    Convert a historical bug record into searchable text.

    More relevant fields are included so that retrieval can match
    both the problem description and the actual code context.
    """

    parts = []

    fields = [
        ("bug_description", None),
        ("error", "Error"),
        ("bug_type", "Type"),
        ("language", "Language"),
        ("buggy_code", "Code"),
        ("solution", "Solution"),
    ]

    for key, label in fields:

        value = record.get(
            key
        )

        if not value:
            continue

        value = str(value)

        if label:

            parts.append(
                f"{label}: {value}"
            )

        else:

            parts.append(
                value
            )

    return "\n".join(
        parts
    )


# ============================================================
# INDEX BUILD
# ============================================================

def _ensure_index_built():
    """
    Build the RAG index automatically if it does not exist.
    """

    vectorizer_exists = os.path.exists(
        TFIDF_VECTORIZER_PATH
    )

    metadata_exists = os.path.exists(
        METADATA_PATH
    )

    if (
        vectorizer_exists
        and metadata_exists
    ):
        return

    print(
        "[retriever] RAG index not found. "
        "Building it now..."
    )

    from rag.build_index import build

    build()


# ============================================================
# LOAD INDEX
# ============================================================

def _load():
    """
    Load the metadata and TF-IDF vectorizer only once.

    Thread-safe so multiple scan workers cannot attempt to load
    the index simultaneously.
    """

    global _metadata
    global _vectorizer
    global _document_matrix

    if _metadata is not None:
        return

    with _load_lock:

        if _metadata is not None:
            return

        try:

            _ensure_index_built()

        except Exception as exc:

            print(
                "[retriever] Unable to build RAG index: "
                f"{exc}"
            )

            _metadata = []

            _vectorizer = None

            _document_matrix = None

            return

        try:

            import joblib

            with open(
                METADATA_PATH,
                "r",
                encoding="utf-8",
            ) as handle:

                _metadata = json.load(
                    handle
                )

            if not isinstance(
                _metadata,
                list,
            ):

                print(
                    "[retriever] Invalid metadata format."
                )

                _metadata = []

                return

            if not _metadata:

                print(
                    "[retriever] RAG metadata is empty."
                )

                _vectorizer = None

                _document_matrix = None

                return

            _vectorizer = joblib.load(
                TFIDF_VECTORIZER_PATH
            )

            documents = [
                _embedding_text(
                    record
                )
                for record in _metadata
            ]

            _document_matrix = (
                _vectorizer.transform(
                    documents
                )
            )

            print(
                "[retriever] TF-IDF RAG index ready: "
                f"{len(_metadata)} historical records."
            )

        except Exception as exc:

            print(
                "[retriever] Failed to load "
                f"TF-IDF index: {exc}"
            )

            _metadata = []

            _vectorizer = None

            _document_matrix = None


# ============================================================
# QUERY TEXT
# ============================================================

def _build_query(
    query_text: str,
):
    """
    Normalize the query before retrieval.
    """

    if not query_text:
        return ""

    return str(
        query_text
    ).strip()


# ============================================================
# RETRIEVE SIMILAR BUGS
# ============================================================

def retrieve_similar_bugs(
    query_text: str,
    top_k: int = 3,
):
    """
    Retrieve the most similar historical bugs.

    Parameters
    ----------
    query_text:
        Text describing the current bug.

    top_k:
        Maximum number of historical records to return.

    Returns
    -------
    list
        Example:

        [
            {
                "record": {...},
                "similarity": 0.82
            }
        ]
    """

    query = _build_query(
        query_text
    )

    if not query:
        return []

    _load()

    if (
        not _metadata
        or _vectorizer is None
        or _document_matrix is None
    ):

        return []

    try:

        from sklearn.metrics.pairwise import (
            cosine_similarity
        )

        query_vector = (
            _vectorizer.transform(
                [query]
            )
        )

        scores = cosine_similarity(
            query_vector,
            _document_matrix,
        )[0]

        requested_k = max(
            1,
            int(top_k),
        )

        result_count = min(
            requested_k,
            len(scores),
        )

        indices = scores.argsort()[
            ::-1
        ][:result_count]

        results = []

        for index in indices:

            score = float(
                scores[index]
            )

            # Ignore effectively unrelated records.
            if score <= 0.01:
                continue

            record = _metadata[
                int(index)
            ]

            results.append(
                {
                    "record": record,
                    "similarity": min(
                        max(
                            score,
                            0.0,
                        ),
                        1.0,
                    ),
                }
            )

        return results

    except Exception as exc:

        print(
            "[retriever] Retrieval failed: "
            f"{exc}"
        )

        return []


# ============================================================
# RAG QUERY BUILDER
# ============================================================

def build_finding_query(
    finding: dict,
):
    """
    Build a richer RAG query from a detector finding.

    Instead of searching only for the error message, include:
        - bug type
        - rule
        - cause
        - file language
        - current code

    This gives TF-IDF more useful information to match against
    historical bugs.
    """

    parts = []

    file_path = finding.get(
        "file",
        "",
    )

    bug_type = finding.get(
        "bug_type",
        "",
    )

    rule = finding.get(
        "rule",
        "",
    )

    error = finding.get(
        "error",
        "",
    )

    cause = finding.get(
        "cause",
        "",
    )

    current_code = finding.get(
        "current_code",
        "",
    )

    if file_path:

        parts.append(
            f"File: {file_path}"
        )

    if bug_type:

        parts.append(
            f"Bug type: {bug_type}"
        )

    if rule:

        parts.append(
            f"Detection rule: {rule}"
        )

    if error:

        parts.append(
            f"Error: {error}"
        )

    if cause:

        parts.append(
            f"Cause: {cause}"
        )

    if current_code:

        parts.append(
            f"Code: {current_code}"
        )

    return "\n".join(
        parts
    )


# ============================================================
# CONVENIENCE RETRIEVAL
# ============================================================

def retrieve_for_finding(
    finding: dict,
    top_k: int = 3,
):
    """
    Retrieve historical bugs specifically for one finding.
    """

    query = build_finding_query(
        finding
    )

    return retrieve_similar_bugs(
        query,
        top_k=top_k,
    )


# ============================================================
# INDEX STATUS
# ============================================================

def get_retriever_status():
    """
    Return lightweight information about the RAG index.

    Useful for health/debugging endpoints.
    """

    _load()

    return {
        "index_type": "TF-IDF",
        "loaded": (
            _vectorizer is not None
            and _document_matrix is not None
        ),
        "records": (
            len(_metadata)
            if _metadata
            else 0
        ),
    }
