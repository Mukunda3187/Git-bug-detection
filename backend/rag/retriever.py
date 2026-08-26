"""
Loads the RAG index and metadata. Supports both:
1. FAISS + semantic embeddings (if sentence-transformers available)
2. TF-IDF fallback (lightweight, works everywhere)

If the index hasn't been built yet, this module builds it automatically
from whatever is in datasets/normalized/.
"""
import json
import os

INDEX_DIR = os.path.join(os.path.dirname(__file__), "index")
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
TFIDF_VECTORIZER_PATH = os.path.join(INDEX_DIR, "vectorizer.joblib")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.json")

_index = None
_metadata = None
_model = None
_vectorizer = None
_use_semantic = False

# Semantic embeddings are disabled on Render.
# We use the lightweight TF-IDF retriever instead.
_SEMANTIC_AVAILABLE = False


def _ensure_index_built():
    """Build index if it doesn't exist."""
    if not os.path.exists(METADATA_PATH):
        print("[retriever] Index not found. Building from datasets...")
        try:
            from rag.build_index import build
            build()
        except Exception as e:
            print(f"[retriever] Failed to build index: {e}")


def _load():
    """Load index, metadata, and model once."""
    global _index, _metadata, _model, _vectorizer, _use_semantic
    
    if _metadata is not None:  # Already loaded
        return
    
    _ensure_index_built()
    
    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            _metadata = json.load(f)
    except Exception as e:
        print(f"[retriever] ❌ Failed to load metadata: {e}")
        _metadata = []
        return
    
    # Try semantic embeddings (FAISS + sentence-transformers)
    if _SEMANTIC_AVAILABLE and os.path.exists(INDEX_PATH):
        try:
            _model = SentenceTransformer("all-MiniLM-L6-v2")
            _index = faiss.read_index(INDEX_PATH)
            _use_semantic = True
            print(f"[retriever] ✅ Loaded FAISS semantic index with {_index.ntotal} records")
            return
        except Exception as e:
            print(f"[retriever] ⚠️  Semantic embeddings unavailable: {e}. Falling back to TF-IDF.")
    
    # Fallback to TF-IDF (lightweight)
    try:
        import joblib
        _vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
        _use_semantic = False
        print(f"[retriever] ✅ Loaded TF-IDF index with {len(_metadata)} records")
    except Exception as e:
        print(f"[retriever] ⚠️  TF-IDF also unavailable: {e}")
        _vectorizer = None


def retrieve_similar_bugs(query_text: str, top_k: int = 3):
    """
    Retrieve the top-k similar bugs from the knowledge base.
    
    Uses semantic embeddings if available, otherwise falls back to TF-IDF.
    
    Args:
        query_text: The bug description/code to search for
        top_k: Number of results to return
    
    Returns:
        List of dicts with 'record' (metadata) and 'similarity' (score)
    """
    _load()
    
    if _metadata is None or len(_metadata) == 0:
        print("[retriever] ⚠️  No metadata available")
        return []
    
    try:
        if _use_semantic and _model is not None and _index is not None:
            return _retrieve_semantic(query_text, top_k)
        elif _vectorizer is not None:
            return _retrieve_tfidf(query_text, top_k)
        else:
            print("[retriever] ❌ No retrieval method available")
            return []
    except Exception as e:
        print(f"[retriever] ❌ Error during retrieval: {e}")
        return []


def _retrieve_semantic(query_text: str, top_k: int):
    """Retrieve using FAISS semantic embeddings."""
    try:
        import numpy as np
        
        # Generate embedding for query
        query_embedding = _model.encode([query_text], convert_to_numpy=True)
        
        # Search FAISS index
        k = min(top_k, _index.ntotal)
        distances, indices = _index.search(query_embedding.astype("float32"), k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            # Convert L2 distance to similarity score
            similarity = max(0, 1.0 / (1.0 + float(distance)))
            
            # Filter out very low similarity matches
            if similarity >= 0.1:
                try:
                    record = _metadata[int(idx)]
                    results.append({
                        "record": record,
                        "similarity": similarity,
                    })
                except (IndexError, TypeError):
                    continue
        
        return results
    except Exception as e:
        print(f"[retriever] Semantic retrieval failed: {e}")
        return []


def _retrieve_tfidf(query_text: str, top_k: int):
    """Retrieve using TF-IDF (lightweight fallback)."""
    try:
        from sklearn.metrics.pairwise import cosine_similarity

        query_vec = _vectorizer.transform([query_text])
        document_matrix = _vectorizer.transform(
            [_embedding_text(r) for r in _metadata]
        )
        scores = cosine_similarity(query_vec, document_matrix)[0]

        top_k = min(top_k, len(scores))
        top_indices = scores.argsort()[::-1][:top_k]

        results = []

        for idx in top_indices:
            score = float(scores[idx])

            if score > 0.01:
                results.append({
                    "record": _metadata[idx],
                    "similarity": min(score, 1.0),
                })

        return results

    except Exception as e:
        print(f"[retriever] TF-IDF retrieval failed: {e}")
        return []


def _embedding_text(record: dict) -> str:
    """Extract text from a record for embedding."""
    parts = []
    if record.get("bug_description"):
        parts.append(record["bug_description"])
    if record.get("error"):
        parts.append(f"Error: {record['error']}")
    if record.get("bug_type"):
        parts.append(f"Type: {record['bug_type']}")
    if record.get("buggy_code"):
        parts.append(f"Code: {record['buggy_code']}")
    return "\n".join(filter(None, parts))
