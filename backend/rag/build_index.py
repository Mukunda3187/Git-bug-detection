"""
Reads every *.jsonl file in datasets/normalized/, converts each entry into
a TF-IDF vector, and saves the vectorizer + vectors to disk.

Why TF-IDF instead of a neural embedding model: sentence-transformers pulls
in PyTorch, which needs hundreds of MB of RAM just to import and also needs
to download the model from Hugging Face on first use - both are unreliable
on a free-tier host with limited memory and no guarantee of outbound access
at runtime. TF-IDF has neither problem: it installs light, builds instantly,
and works fully offline.

Run this once after adding/updating any normalized dataset:
    python -m rag.build_index

Produces:
    rag/index/vectorizer.joblib   -> the fitted TF-IDF vectorizer
    rag/index/metadata.json       -> the original records, same row order as the vectorizer's vocabulary
"""
import glob
import json
import os

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

DATASETS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "normalized")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "index")


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
                    if not record.get("bug_description") or not record.get("buggy_code"):
                        print(f"[build_index] Skipping malformed record in {os.path.basename(path)}:{line_num} - missing bug_description or buggy_code")
                        continue
                    records.append(record)
                except json.JSONDecodeError as e:
                    print(f"[build_index] Skipping invalid JSON in {os.path.basename(path)}:{line_num}: {e}")
                    continue
    return records


def embedding_text(record: dict) -> str:
    """Extract text from a record for embedding - keep in sync with retriever.py's _embedding_text."""
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


def build():
    """Build a TF-IDF index from normalized datasets."""
    records = load_all_records()

    if not records:
        print(f"[build_index] No valid normalized dataset files found in {DATASETS_DIR}. "
              f"Run the normalize_*.py scripts in datasets/ first (or keep "
              f"the bundled sample.jsonl to test the pipeline).")
        return

    print(f"[build_index] Loaded {len(records)} valid records. Fitting TF-IDF vectorizer...")
    texts = [embedding_text(r) for r in records]

    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    vectorizer.fit(texts)  # the vectorizer itself is saved; vectors are recomputed cheaply at query time

    os.makedirs(INDEX_DIR, exist_ok=True)

    try:
        joblib.dump(vectorizer, os.path.join(INDEX_DIR, "vectorizer.joblib"))
        with open(os.path.join(INDEX_DIR, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(records, f)
        print(f"[build_index] Index built successfully!")
        print(f"   - Vectorizer: {os.path.join(INDEX_DIR, 'vectorizer.joblib')}")
        print(f"   - Metadata: {os.path.join(INDEX_DIR, 'metadata.json')}")
        print(f"   - Total records: {len(records)}")
    except Exception as e:
        print(f"[build_index] Failed to save index: {e}")
        return


if __name__ == "__main__":
    build()
