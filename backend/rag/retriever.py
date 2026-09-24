
"""Memory-conscious TF-IDF-only retrieval for the RAG knowledge base."""

import json
import os
import threading

INDEX_DIR = os.path.join(os.path.dirname(__file__), "index")
TFIDF_VECTORIZER_PATH = os.path.join(INDEX_DIR, "vectorizer.joblib")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.json")

_metadata = None
_vectorizer = None
_document_matrix = None
_load_lock = threading.Lock()


def _embedding_text(record: dict) -> str:
    parts = []

    for key, label in (
        ("bug_description", None),
        ("error", "Error"),
        ("bug_type", "Type"),
        ("language", "Language"),
        ("buggy_code", "Code"),
        ("solution", "Solution"),
    ):
        value = record.get(key)
        if value:
            parts.append(
                str(value) if label is None else f"{label}: {value}"
            )

    return "\n".join(parts)


def _ensure_index_built():
    if (
        os.path.exists(TFIDF_VECTORIZER_PATH)
        and os.path.exists(METADATA_PATH)
    ):
        return

    from rag.build_index import build
    build()


def _load():
    global _metadata, _vectorizer, _document_matrix

    if _metadata is not None:
        return

    with _load_lock:
        if _metadata is not None:
            return

        _ensure_index_built()

        try:
            import joblib

            with open(METADATA_PATH, "r", encoding="utf-8") as handle:
                _metadata = json.load(handle)

            if not _metadata:
                _vectorizer = None
                _document_matrix = None
                return

            _vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)

            _document_matrix = _vectorizer.transform(
                [_embedding_text(record) for record in _metadata]
            )

            print(
                f"[retriever] TF-IDF retrieval ready: "
                f"{len(_metadata)} records."
            )

        except Exception as exc:
            print(f"[retriever] Failed to load TF-IDF index: {exc}")
            _metadata = _metadata or []
            _vectorizer = None
            _document_matrix = None


def retrieve_similar_bugs(query_text: str, top_k: int = 3):
    _load()

    if (
        not _metadata
        or _vectorizer is None
        or _document_matrix is None
        or not query_text
    ):
        return []

    try:
        from sklearn.metrics.pairwise import cosine_similarity

        query_vec = _vectorizer.transform([query_text])
        scores = cosine_similarity(query_vec, _document_matrix)[0]

        count = min(max(1, int(top_k)), len(scores))
        indices = scores.argsort()[::-1][:count]

        results = []

        for idx in indices:
            score = float(scores[idx])

            if score <= 0.01:
                continue

            results.append({
                "record": _metadata[int(idx)],
                "similarity": min(score, 1.0),
            })

        return results

    except Exception as exc:
        print(f"[retriever] TF-IDF retrieval error: {exc}")
        return []
