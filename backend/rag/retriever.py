"""
Loads the FAISS index and metadata built by build_index.py and exposes
a single function: retrieve_similar_bugs(query_text, top_k).

Uses semantic embeddings via sentence-transformers for better bug matching.

If the index hasn't been built yet, this module builds it automatically
from whatever is in datasets/normalized/ (so a fresh clone works immediately
using the bundled sample.jsonl).
"""
import json
import os

from sentence_transformers import SentenceTransformer

INDEX_DIR = os.path.join(os.path.dirname(__file__), "index")
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.json")

MODEL_NAME = "all-MiniLM-L6-v2"

_index = None
_metadata = None
_model = None


def _ensure_index_built():
    """Build index if it doesn't exist."""
    if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
        print("[retriever] Index not found. Building from datasets...")
        try:
            from rag.build_index import build
            build()
        except Exception as e:
            print(f"[retriever] Failed to build index: {e}")


def _load():
    """Load FAISS index, metadata, and model once."""
    global _index, _metadata, _model
    
    if _index is not None and _metadata is not None and _model is not None:
        return
    
    _ensure_index_built()
    
    try:
        import faiss
    except ImportError:
        print("[retriever] ❌ FAISS not installed. Install with: pip install faiss-cpu")
        return
    
    try:
        _model = SentenceTransformer(MODEL_NAME)
        _index = faiss.read_index(INDEX_PATH)
        
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            _metadata = json.load(f)
        
        print(f"[retriever] ✅ Loaded FAISS index with {_index.ntotal} records")
    except Exception as e:
        print(f"[retriever] ❌ Failed to load index/model: {e}")
        _index = None
        _metadata = None
        _model = None


def retrieve_similar_bugs(query_text: str, top_k: int = 3):
    """
    Retrieve the top-k similar bugs from the knowledge base.
    
    Args:
        query_text: The bug description/code to search for
        top_k: Number of results to return
    
    Returns:
        List of dicts with 'record' (metadata) and 'similarity' (score)
    """
    _load()
    
    if _index is None or _metadata is None or _model is None:
        print("[retriever] ⚠️  Index not available, returning empty results")
        return []
    
    if _index.ntotal == 0:
        return []
    
    try:
        # Generate embedding for query
        query_embedding = _model.encode([query_text], convert_to_numpy=True)
        
        # Search FAISS index (returns distances and indices)
        # L2 distance: smaller = more similar
        k = min(top_k, _index.ntotal)
        distances, indices = _index.search(query_embedding.astype("float32"), k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            # Convert L2 distance to similarity score (0-1 range)
            # Use exponential decay: similarity = exp(-distance)
            similarity = max(0, 1.0 / (1.0 + distance))
            
            # Filter out very low similarity matches
            if similarity >= 0.1:  # Configurable threshold
                try:
                    record = _metadata[int(idx)]
                    results.append({
                        "record": record,
                        "similarity": float(similarity),
                    })
                except (IndexError, TypeError) as e:
                    print(f"[retriever] ⚠️  Error retrieving record at index {idx}: {e}")
                    continue
        
        return results
    
    except Exception as e:
        print(f"[retriever] ❌ Error during retrieval: {e}")
        return []
