"""
Loads the FAISS index built by build_index.py and exposes a single
function: retrieve_similar_bugs(query_text, top_k).

If the index hasn't been built yet, this module builds it automatically
from whatever is in datasets/normalized/ (so a fresh clone of the project
works immediately using the bundled sample.jsonl).
"""
import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_DIR = os.path.join(os.path.dirname(__file__), "index")
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.json")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

_model = None
_index = None
_metadata = None


def _ensure_index_built():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
        from rag.build_index import build
        build()


def _load():
    global _model, _index, _metadata
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    if _index is None or _metadata is None:
        _ensure_index_built()
        _index = faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            _metadata = json.load(f)


def retrieve_similar_bugs(query_text: str, top_k: int = 3):
    """
    Returns a list of dicts: {record, similarity} sorted by similarity
    descending. similarity is a 0-1 cosine similarity score.
    """
    _load()
    if _index.ntotal == 0:
        return []

    query_vec = _model.encode([query_text], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vec)

    top_k = min(top_k, _index.ntotal)
    scores, indices = _index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append({
            "record": _metadata[idx],
            "similarity": float(score),
        })
    return results
