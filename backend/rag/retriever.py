"""
Loads the TF-IDF vectorizer + vectors built by build_index.py and exposes
a single function: retrieve_similar_bugs(query_text, top_k).

If the index hasn't been built yet, this module builds it automatically
from whatever is in datasets/normalized/ (so a fresh clone of the project
works immediately using the bundled sample.jsonl).
"""
import json
import os

import joblib
from sklearn.metrics.pairwise import cosine_similarity

INDEX_DIR = os.path.join(os.path.dirname(__file__), "index")
VECTORIZER_PATH = os.path.join(INDEX_DIR, "vectorizer.joblib")
VECTORS_PATH = os.path.join(INDEX_DIR, "vectors.joblib")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.json")

_vectorizer = None
_vectors = None
_metadata = None


def _ensure_index_built():
    if not os.path.exists(VECTORIZER_PATH) or not os.path.exists(VECTORS_PATH) or not os.path.exists(METADATA_PATH):
        from rag.build_index import build
        build()


def _load():
    global _vectorizer, _vectors, _metadata
    if _vectorizer is None or _vectors is None or _metadata is None:
        _ensure_index_built()
        _vectorizer = joblib.load(VECTORIZER_PATH)
        _vectors = joblib.load(VECTORS_PATH)
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            _metadata = json.load(f)


def retrieve_similar_bugs(query_text: str, top_k: int = 3):
    _load()
    if _vectors.shape[0] == 0:
        return []

    query_vec = _vectorizer.transform([query_text])
    scores = cosine_similarity(query_vec, _vectors)[0]

    top_k = min(top_k, len(scores))
    top_indices = scores.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] <= 0:
            continue
        results.append({
            "record": _metadata[idx],
            "similarity": float(scores[idx]),
        })
    return results
