"""
RAG retriever with semantic embedding support.

Primary retrieval:
    Sentence Transformer embeddings

Model:
    all-MiniLM-L6-v2

Semantic embeddings:
    rag/index/embeddings.npy

Metadata:
    rag/index/metadata.json

Fallback:
    TF-IDF
"""

import json
import os
import threading

import numpy as np


INDEX_DIR = os.path.join(
    os.path.dirname(__file__),
    "index",
)

EMBEDDINGS_PATH = os.path.join(
    INDEX_DIR,
    "embeddings.npy",
)

TFIDF_VECTORIZER_PATH = os.path.join(
    INDEX_DIR,
    "vectorizer.joblib",
)

METADATA_PATH = os.path.join(
    INDEX_DIR,
    "metadata.json",
)

MODEL_NAME = "all-MiniLM-L6-v2"


_metadata = None
_model = None
_embeddings = None

_vectorizer = None
_document_matrix = None

_use_semantic = False

_load_lock = threading.Lock()


def _ensure_index_built():
    """Build the RAG index if metadata or embeddings are missing."""

    if (
        not os.path.exists(METADATA_PATH)
        or not os.path.exists(EMBEDDINGS_PATH)
    ):
        print(
            "[retriever] Semantic index not found. "
            "Building from datasets..."
        )

        try:
            from rag.build_index import build

            build()

        except Exception as e:
            print(
                f"[retriever] Failed to build index: {e}"
            )


def _load():
    """
    Load metadata, semantic model, embeddings and TF-IDF fallback.

    Loading is performed once and protected by a lock because multiple
    bug-analysis threads may request retrieval at the same time.
    """

    global _metadata
    global _model
    global _embeddings
    global _vectorizer
    global _document_matrix
    global _use_semantic

    if _metadata is not None:
        return

    with _load_lock:

        if _metadata is not None:
            return

        _ensure_index_built()

        # ---------------------------------------------------------
        # Load metadata
        # ---------------------------------------------------------

        try:

            with open(
                METADATA_PATH,
                "r",
                encoding="utf-8",
            ) as f:

                _metadata = json.load(f)

        except Exception as e:

            print(
                f"[retriever] Failed to load metadata: {e}"
            )

            _metadata = []

            return

        if not _metadata:

            print(
                "[retriever] No metadata records available."
            )

            return

        # ---------------------------------------------------------
        # Try semantic embeddings
        # ---------------------------------------------------------

        if os.path.exists(EMBEDDINGS_PATH):

            try:

                from sentence_transformers import SentenceTransformer

                print(
                    f"[retriever] Loading semantic model: "
                    f"{MODEL_NAME}"
                )

                _model = SentenceTransformer(
                    MODEL_NAME
                )

                _embeddings = np.load(
                    EMBEDDINGS_PATH
                )

                _embeddings = _embeddings.astype(
                    "float32"
                )

                if len(_embeddings) != len(_metadata):

                    raise ValueError(
                        "Number of embeddings does not match "
                        "number of metadata records."
                    )

                _use_semantic = True

                print(
                    "[retriever] Semantic embeddings loaded successfully."
                )

                print(
                    f"[retriever] Embedding records: "
                    f"{len(_embeddings)}"
                )

                print(
                    f"[retriever] Embedding dimension: "
                    f"{_embeddings.shape[1]}"
                )

                return

            except Exception as e:

                print(
                    "[retriever] Semantic embeddings unavailable: "
                    f"{e}"
                )

                _model = None
                _embeddings = None
                _use_semantic = False

        # ---------------------------------------------------------
        # TF-IDF fallback
        # ---------------------------------------------------------

        try:

            import joblib

            _vectorizer = joblib.load(
                TFIDF_VECTORIZER_PATH
            )

            _document_matrix = _vectorizer.transform(
                [
                    _embedding_text(record)
                    for record in _metadata
                ]
            )

            _use_semantic = False

            print(
                f"[retriever] Loaded TF-IDF fallback with "
                f"{len(_metadata)} records."
            )

        except Exception as e:

            print(
                f"[retriever] TF-IDF also unavailable: {e}"
            )

            _vectorizer = None
            _document_matrix = None


def retrieve_similar_bugs(
    query_text: str,
    top_k: int = 3,
):
    """
    Retrieve the most similar historical bugs.

    Semantic embeddings are used when available.
    TF-IDF is used as a fallback.
    """

    _load()

    if not _metadata:

        print(
            "[retriever] No metadata available."
        )

        return []

    try:

        if (
            _use_semantic
            and _model is not None
            and _embeddings is not None
        ):

            return _retrieve_semantic(
                query_text,
                top_k,
            )

        if (
            _vectorizer is not None
            and _document_matrix is not None
        ):

            return _retrieve_tfidf(
                query_text,
                top_k,
            )

        print(
            "[retriever] No retrieval method available."
        )

        return []

    except Exception as e:

        print(
            f"[retriever] Error during retrieval: {e}"
        )

        return []


def _retrieve_semantic(
    query_text: str,
    top_k: int,
):
    """
    Retrieve bugs using semantic embeddings.

    The embeddings are normalized, so cosine similarity can be
    calculated using a simple dot product.
    """

    try:

        query_embedding = _model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        query_embedding = query_embedding.astype(
            "float32"
        )

        scores = np.dot(
            _embeddings,
            query_embedding[0],
        )

        top_k = min(
            top_k,
            len(scores),
        )

        top_indices = np.argsort(
            scores
        )[::-1][:top_k]

        results = []

        for idx in top_indices:

            score = float(
                scores[idx]
            )

            # Cosine similarity after normalized embeddings
            # should normally be between -1 and 1.
            similarity = max(
                0.0,
                min(
                    score,
                    1.0,
                ),
            )

            if similarity >= 0.10:

                results.append(
                    {
                        "record": _metadata[
                            int(idx)
                        ],
                        "similarity": similarity,
                    }
                )

        print(
            f"[retriever] Semantic search returned "
            f"{len(results)} results."
        )

        return results

    except Exception as e:

        print(
            f"[retriever] Semantic retrieval failed: {e}"
        )

        return []


def _retrieve_tfidf(
    query_text: str,
    top_k: int,
):
    """Retrieve using TF-IDF fallback."""

    try:

        from sklearn.metrics.pairwise import cosine_similarity

        query_vec = _vectorizer.transform(
            [query_text]
        )

        scores = cosine_similarity(
            query_vec,
            _document_matrix,
        )[0]

        top_k = min(
            top_k,
            len(scores),
        )

        top_indices = scores.argsort()[
            ::-1
        ][:top_k]

        results = []

        for idx in top_indices:

            score = float(
                scores[idx]
            )

            if score > 0.01:

                results.append(
                    {
                        "record": _metadata[
                            int(idx)
                        ],
                        "similarity": min(
                            score,
                            1.0,
                        ),
                    }
                )

        return results

    except Exception as e:

        print(
            f"[retriever] TF-IDF retrieval failed: {e}"
        )

        return []


def _embedding_text(
    record: dict,
) -> str:
    """
    Create the same text representation used by build_index.py.
    """

    parts = []

    if record.get("bug_description"):

        parts.append(
            str(
                record["bug_description"]
            )
        )

    if record.get("error"):

        parts.append(
            f"Error: {record['error']}"
        )

    if record.get("bug_type"):

        parts.append(
            f"Type: {record['bug_type']}"
        )

    if record.get("language"):

        parts.append(
            f"Language: {record['language']}"
        )

    if record.get("buggy_code"):

        parts.append(
            f"Code: {record['buggy_code']}"
        )

    return "\n".join(
        filter(
            None,
            parts,
        )
    )
