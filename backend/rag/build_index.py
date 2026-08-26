"""
Reads every *.jsonl file in datasets/normalized/, converts each entry into
a semantic vector using sentence-transformers, and saves to a FAISS index.

This uses actual neural embeddings (all-MiniLM-L6-v2) for much better semantic
matching than TF-IDF, while keeping memory usage reasonable (~200MB for large datasets).

Run this once after adding/updating any normalized dataset:
    python -m rag.build_index

Produces:
    rag/index/faiss.index      -> the FAISS index
    rag/index/metadata.json    -> the original records, same order as index
"""
import glob
import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer

DATASETS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "normalized")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "index")
MODEL_NAME = "all-MiniLM-L6-v2"  # Fast, small, effective for bug retrieval


def load_all_records():
    """Load and validate all records from normalized datasets."""
    records = []
    for path in glob.glob(os.path.join(DATASETS_DIR, "*.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    # Validate required fields
                    if not record.get("bug_description") or not record.get("buggy_code"):
                        print(f"⚠️  Skipping malformed record in {os.path.basename(path)}:{line_num} - missing bug_description or buggy_code")
                        continue
                    records.append(record)
                except json.JSONDecodeError as e:
                    print(f"⚠️  Skipping invalid JSON in {os.path.basename(path)}:{line_num}: {e}")
                    continue
    return records


def embedding_text(record: dict) -> str:
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
    if record.get("solution"):
        parts.append(f"Solution: {record['solution']}")
    return "\n".join(filter(None, parts))


def build():
    """Build FAISS index from normalized datasets."""
    records = load_all_records()
    
    if not records:
        print(f"⚠️  No valid normalized dataset files found in {DATASETS_DIR}. "
              f"Run the normalize_*.py scripts in datasets/ first (or keep "
              f"the bundled sample.jsonl to test the pipeline).")
        return

    print(f"✅ Loaded {len(records)} valid records. Loading embedding model '{MODEL_NAME}'...")
    
    try:
        model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("Make sure sentence-transformers is installed: pip install sentence-transformers")
        return

    print(f"Generating embeddings for {len(records)} records...")
    texts = [embedding_text(r) for r in records]
    
    try:
        embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    except Exception as e:
        print(f"❌ Failed to generate embeddings: {e}")
        return

    # Create FAISS index
    try:
        import faiss
    except ImportError:
        print("❌ FAISS not installed. Install it with: pip install faiss-cpu")
        return

    print(f"Building FAISS index with {len(embeddings)} vectors of dimension {embeddings.shape[1]}...")
    
    # Use IndexFlatL2 for simplicity (exact search with L2 distance)
    # For large datasets (>100k), consider IndexIVFFlat for faster approximate search
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))

    os.makedirs(INDEX_DIR, exist_ok=True)
    
    try:
        faiss.write_index(index, os.path.join(INDEX_DIR, "faiss.index"))
        with open(os.path.join(INDEX_DIR, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(records, f)
        print(f"✅ Index built successfully!")
        print(f"   - FAISS index: {os.path.join(INDEX_DIR, 'faiss.index')}")
        print(f"   - Metadata: {os.path.join(INDEX_DIR, 'metadata.json')}")
        print(f"   - Total records: {len(records)}")
        print(f"   - Embedding dimension: {embeddings.shape[1]}")
    except Exception as e:
        print(f"❌ Failed to save index: {e}")
        return


if __name__ == "__main__":
    build()
