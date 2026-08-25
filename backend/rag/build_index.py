"""
Reads every *.jsonl file in datasets/normalized/, embeds each entry with a
local sentence-transformers model, and builds a FAISS index.

Run this once after adding/updating any normalized dataset:
    python -m rag.build_index

Produces:
    rag/index/faiss.index      -> the vector index
    rag/index/metadata.json    -> the original records, in the same order
                                   as the vectors, so a FAISS result index
                                   can be mapped straight back to a record
"""
import glob
import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DATASETS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "normalized")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "index")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # small, free, runs on CPU


def load_all_records():
    records = []
    for path in glob.glob(os.path.join(DATASETS_DIR, "*.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    return records


def embedding_text(record: dict) -> str:
    parts = []
    if record.get("bug_description"):
        parts.append(record["bug_description"])
    if record.get("error"):
        parts.append(f"Error: {record['error']}")
    parts.append(record.get("buggy_code", ""))
    return "\n".join(parts)


def build():
    records = load_all_records()
    if not records:
        print(f"No normalized dataset files found in {DATASETS_DIR}. "
              f"Run the normalize_*.py scripts in datasets/ first (or keep "
              f"the bundled sample.jsonl to test the pipeline).")
        return

    print(f"Loaded {len(records)} records. Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts = [embedding_text(r) for r in records]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)  # so inner product == cosine similarity

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(INDEX_DIR, "faiss.index"))
    with open(os.path.join(INDEX_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(records, f)

    print(f"Index built with {index.ntotal} vectors -> {INDEX_DIR}")


if __name__ == "__main__":
    build()
