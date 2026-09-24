"""
Loads the RAG index and metadata. Supports both:
1. FAISS + semantic embeddings (if sentence-transformers AND faiss are
   installed - both are optional extras, not in requirements.txt)
2. TF-IDF fallback (lightweight, works everywhere - this is what actually
   runs by default, see build_index.py for why)

If the index hasn't been built yet, this module builds it automatically
from whatever is in datasets/normalized/.
"""
import json
import os
import threading

INDEX_DIR = os.path.join(os.path.dirname(__file__), "index")
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
TFIDF_VECTORIZER_PATH = os.path.join(INDEX_DIR, "vectorizer.joblib")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.json")

_index = None
_metadata = None
_model = None
_vectorizer = None
_document_matrix = None  # TF-IDF vectors for every record in _metadata, computed
                          # once in _load() instead of on every single retrieval
                          # call - see _retrieve_tfidf() below for why that
                          # re-transform was there before and why it's gone now.
_use_semantic = False

# Guards the one-time loading/building work in _load() below. Without this,
# several threads calling retrieve_similar_bugs() for the first time at once
# (e.g. a scan's first few findings, now analyzed concurrently) could all see
# _metadata as None simultaneously and race into _ensure_index_built() /
# joblib.load() together - at best wasted duplicate work, at worst more than
# one thread writing the index files at the same time on a cold start.
_load_lock = threading.Lock()

# Semantic embeddings are disabled by default (they need sentence-transformers
# + faiss, which aren't in requirements.txt, and are unreliable on a
# free-tier host - see build_index.py's docstring). The lightweight TF-IDF
# retriever is used instead. Flip this to True only after adding both
# packages to requirements.txt - both SentenceTransformer and faiss are
# imported lazily inside _load()/_retrieve_semantic() below precisely so
# this flag is safe to toggle without a NameError if the packages are
# actually installed.
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
    """Load index, metadata, and model once - thread-safe (see _load_lock above)."""
    global _index, _metadata, _model, _vectorizer, _document_matrix, _use_semantic

    if _metadata is not None:  # Already loaded - fast path, no lock needed
        return

    with _load_lock:
        if _metadata is not None:  # Double-checked: another thread may have
            return                  # finished loading while this one waited

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
            # Precomputed once, here, instead of inside _retrieve_tfidf() -
            # every retrieval call used to re-run transform() over the whole
            # dataset from scratch, which is pure wasted repeated work: the
            # dataset doesn't change between calls, so its vectors don't
            # either. For a scan with many findings this was doing the same
            # computation over and over for no reason.
            _document_matrix = _vectorizer.transform(
                [_embedding_text(r) for r in _metadata]
            )
            print(f"[retriever] ✅ Loaded TF-IDF index with {len(_metadata)} records")
        except Exception as e:
            print(f"[retriever] ⚠️  TF-IDF also unavailable: {e}")
            _vectorizer = None
            _document_matrix = None


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
        elif _vectorizer is not None and _document_matrix is not None:
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
        scores = cosine_similarity(query_vec, _document_matrix)[0]

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
