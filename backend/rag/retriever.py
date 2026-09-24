"""
RAG retriever.

Retrieval priority:

1. FAISS + Sentence Transformer semantic search
2. TF-IDF fallback

Semantic model:
    all-MiniLM-L6-v2

FAISS index:
    rag/index/faiss.index

Embeddings:
    rag/index/embeddings.npy

Metadata:
    rag/index/metadata.json
"""

import json
import os
import threading

import numpy as np


INDEX_DIR = os.path.join(
    os.path.dirname(__file__),
    "index",
)

FAISS_INDEX_PATH = os.path.join(
    INDEX_DIR,
    "faiss.index",
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
_faiss_index = None
_embeddings = None

_vectorizer = None
_document_matrix = None

_use_semantic = False

_load_lock = threading.Lock()


def _ensure_index_built():
    """
    Build the RAG index if the required files do not exist.
    """

    semantic_index_exists = (
        os.path.exists(
            FAISS_INDEX_PATH
        )
        and os.path.exists(
            METADATA_PATH
        )
    )

    tfidf_index_exists = (
        os.path.exists(
            TFIDF_VECTORIZER_PATH
        )
        and os.path.exists(
            METADATA_PATH
        )
    )

    if (
        not semantic_index_exists
        and not tfidf_index_exists
    ):

        print(
            "[retriever] RAG index not found. "
            "Building index..."
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
    Load metadata, semantic model, FAISS index and
    TF-IDF fallback once.
    """

    global _metadata
    global _model
    global _faiss_index
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

                _metadata = json.load(
                    f
                )

        except Exception as e:

            print(
                f"[retriever] Failed to load metadata: {e}"
            )

            _metadata = []

            return

        if not _metadata:

            print(
                "[retriever] Metadata is empty."
            )

            return

        # ---------------------------------------------------------
        # Try semantic + FAISS
        # ---------------------------------------------------------

        if os.path.exists(
            FAISS_INDEX_PATH
        ):

            try:

                from sentence_transformers import (
                    SentenceTransformer,
                )

                import faiss

                print(
                    f"[retriever] Loading semantic model: "
                    f"{MODEL_NAME}"
                )

                _model = SentenceTransformer(
                    MODEL_NAME
                )

                _faiss_index = faiss.read_index(
                    FAISS_INDEX_PATH
                )

                # Load embeddings only for validation/fallback.
                if os.path.exists(
                    EMBEDDINGS_PATH
                ):

                    _embeddings = np.load(
                        EMBEDDINGS_PATH
                    )

                if (
                    _faiss_index.ntotal
                    != len(_metadata)
                ):

                    raise ValueError(
                        "FAISS index record count does not "
                        "match metadata record count."
                    )

                _use_semantic = True

                print(
                    "[retriever] Semantic + FAISS "
                    "retrieval is READY."
                )

                print(
                    f"[retriever] FAISS records: "
                    f"{_faiss_index.ntotal}"
                )

                return

            except Exception as e:

                print(
                    "[retriever] Semantic/FAISS retrieval "
                    f"unavailable: {e}"
                )

                _model = None
                _faiss_index = None
                _use_semantic = False

        # ---------------------------------------------------------
        # TF-IDF fallback
        # ---------------------------------------------------------

        try:

            import joblib

            _vectorizer = joblib.load(
                TFIDF_VECTORIZER_PATH
            )

            _document_matrix = (
                _vectorizer.transform(
                    [
                        _embedding_text(
                            record
                        )
                        for record in _metadata
                    ]
                )
            )

            _use_semantic = False

            print(
                "[retriever] TF-IDF fallback "
                "retrieval is READY."
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

    Semantic FAISS retrieval is preferred.
    TF-IDF is used if semantic retrieval is unavailable.
    """

    _load()

    if (
        _metadata is None
        or len(_metadata) == 0
    ):

        print(
            "[retriever] No metadata available."
        )

        return []

    try:

        if (
            _use_semantic
            and _model is not None
            and _faiss_index is not None
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
            f"[retriever] Retrieval error: {e}"
        )

        return []


def _retrieve_semantic(
    query_text: str,
    top_k: int,
):
    """
    Retrieve using Sentence Transformer embeddings
    and FAISS.

    Embeddings are normalized, so FAISS IndexFlatIP
    gives cosine similarity.
    """

    try:

        query_embedding = _model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32",
        )

        k = min(
            max(
                1,
                top_k,
            ),
            _faiss_index.ntotal,
        )

        scores, indices = (
            _faiss_index.search(
                query_embedding,
                k,
            )
        )

        results = []

        for idx, score in zip(
            indices[0],
            scores[0],
        ):

            if idx < 0:
                continue

            similarity = float(
                max(
                    0.0,
                    min(
                        float(score),
                        1.0,
                    ),
                )
            )

            if similarity < 0.10:
                continue

            try:

                record = _metadata[
                    int(idx)
                ]

            except (
                IndexError,
                TypeError,
            ):

                continue

            results.append(
                {
                    "record": record,
                    "similarity": similarity,
                }
            )

        print(
            f"[retriever] FAISS semantic search "
            f"returned {len(results)} results."
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
    """
    Lightweight TF-IDF fallback.
    """

    try:

        from sklearn.metrics.pairwise import (
            cosine_similarity,
        )

        query_vec = _vectorizer.transform(
            [query_text]
        )

        scores = cosine_similarity(
            query_vec,
            _document_matrix,
        )[0]

        top_k = min(
            max(
                1,
                top_k,
            ),
            len(scores),
        )

        top_indices = (
            scores.argsort()[::-1][:top_k]
        )

        results = []

        for idx in top_indices:

            score = float(
                scores[idx]
            )

            if score <= 0.01:
                continue

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
    Create the same text representation used when
    building the semantic index.
    """

    parts = []

    if record.get(
        "bug_description"
    ):

        parts.append(
            str(
                record[
                    "bug_description"
                ]
            )
        )

    if record.get(
        "error"
    ):

        parts.append(
            f"Error: {record['error']}"
        )

    if record.get(
        "bug_type"
    ):

        parts.append(
            f"Type: {record['bug_type']}"
        )

    if record.get(
        "language"
    ):

        parts.append(
            f"Language: {record['language']}"
        )

    if record.get(
        "buggy_code"
    ):

        parts.append(
            f"Code: {record['buggy_code']}"
        )

    if record.get(
        "solution"
    ):

        parts.append(
            f"Solution: {record['solution']}"
        )

    return "\n".join(
        filter(
            None,
            parts,
        )
    )
