"""
Reads every *.jsonl file in datasets/normalized/, converts each entry into
a TF-IDF vector, and saves the vectorizer + vectors to disk.

Why TF-IDF instead of a neural embedding model: a real sentence-transformers
model pulls in PyTorch, which alone needs several hundred MB of RAM just to
import - that's too much for a free-tier host (512MB). TF-IDF has no such
dependency, loads instantly, and still does genuine similarity search: it
matches on shared error keywords and code tokens, which is exactly the
signal that matters for retrieving similar bugs.

Run this once after adding/updating any normalized dataset:
    python -m rag.build_index

Produces:
    rag/index/vectorizer.joblib   -> the fitted TF-IDF vectorizer
    rag/index/vectors.joblib      -> the sparse TF-IDF matrix, one row per record
    rag/index/metadata.json       -> the original records, same row order as vectors.joblib
"""
import glob
import json
import os

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

DATASETS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "normalized")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "index")


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

    print(f"Loaded {len(records)} records. Fitting TF-IDF vectorizer...")
    texts = [embedding_text(r) for r in records]

    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    vectors = vectorizer.fit_transform(texts)  # sparse matrix, one row per record

    os.makedirs(INDEX_DIR, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(INDEX_DIR, "vectorizer.joblib"))
    joblib.dump(vectors, os.path.join(INDEX_DIR, "vectors.joblib"))
    with open(os.path.join(INDEX_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(records, f)

    print(f"Index built with {vectors.shape[0]} vectors -> {INDEX_DIR}")


if __name__ == "__main__":
    build()
