"""
Semantic RAG retrieval layer.

P2:
    FastEmbed semantic embeddings

P3:
    FAISS vector database

Supports:
    1. Historical bug dataset
    2. Historical GitHub Issues
    3. Historical GitHub Pull Requests
    4. Previous fixes / PR patches
"""

import json
import os
import threading

import numpy as np


# ============================================================
# PATHS
# ============================================================

INDEX_DIR = os.path.join(
    os.path.dirname(__file__),
    "index",
)

METADATA_PATH = os.path.join(
    INDEX_DIR,
    "metadata.json",
)

EMBEDDINGS_PATH = os.path.join(
    INDEX_DIR,
    "semantic_embeddings.npy",
)

FAISS_INDEX_PATH = os.path.join(
    INDEX_DIR,
    "bugs.faiss",
)


# ============================================================
# EMBEDDING MODEL
# ============================================================

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

_embedding_model = None
_metadata = None
_metadata_embeddings = None
_faiss_index = None

_load_lock = threading.RLock()

_github_embedding_cache = {}


# ============================================================
# TEXT HELPERS
# ============================================================

def _safe_text(value):
    if value is None:
        return ""

    return str(value).strip()


def _embedding_text(record: dict) -> str:

    parts = []

    fields = [
        ("bug_description", None),
        ("error", "Error"),
        ("bug_type", "Type"),
        ("language", "Language"),
        ("buggy_code", "Code"),
        ("solution", "Solution"),
    ]

    for key, label in fields:

        value = record.get(key)

        if not value:
            continue

        value = _safe_text(value)

        if label:
            parts.append(
                f"{label}: {value}"
            )
        else:
            parts.append(value)

    return "\n".join(parts)


def _github_artifact_text(artifact: dict) -> str:

    parts = []

    artifact_type = _safe_text(
        artifact.get("artifact_type")
    )

    source = _safe_text(
        artifact.get("source")
    )

    title = _safe_text(
        artifact.get("title")
    )

    body = _safe_text(
        artifact.get("body")
    )

    state = _safe_text(
        artifact.get("state")
    )

    labels = artifact.get("labels") or []

    comments = _safe_text(
        artifact.get("comments")
    )

    patch = _safe_text(
        artifact.get("patch")
    )

    number = artifact.get("number")

    if source:
        parts.append(
            f"Source: {source}"
        )

    if artifact_type:
        parts.append(
            f"Artifact type: {artifact_type}"
        )

    if number is not None:
        parts.append(
            f"Number: {number}"
        )

    if title:
        parts.append(
            f"Title: {title}"
        )

    if state:
        parts.append(
            f"State: {state}"
        )

    if labels:

        clean_labels = [
            str(label)
            for label in labels
            if label
        ]

        if clean_labels:
            parts.append(
                "Labels: "
                + ", ".join(clean_labels)
            )

    if body:
        parts.append(
            f"Description: {body}"
        )

    if comments:
        parts.append(
            f"Comments: {comments}"
        )

    if patch:
        parts.append(
            "Previous fix / patch:\n"
            + patch[:12000]
        )

    if not parts:

        existing_text = _safe_text(
            artifact.get("text")
        )

        if existing_text:
            parts.append(existing_text)

    return "\n".join(parts)


# ============================================================
# FASTEMBED MODEL
# ============================================================

def _get_embedding_model():

    global _embedding_model

    if _embedding_model is not None:
        return _embedding_model

    with _load_lock:

        if _embedding_model is not None:
            return _embedding_model

        try:

            from fastembed import TextEmbedding

            print(
                "[retriever] Loading semantic embedding model: "
                f"{EMBEDDING_MODEL_NAME}"
            )

            _embedding_model = TextEmbedding(
                model_name=EMBEDDING_MODEL_NAME
            )

            print(
                "[retriever] Semantic embedding model ready."
            )

            return _embedding_model

        except Exception as exc:

            print(
                "[retriever] Failed to load semantic "
                f"embedding model: {exc}"
            )

            return None


# ============================================================
# EMBEDDING
# ============================================================

def _encode_texts(texts):

    if not texts:

        return np.empty(
            (0, 0),
            dtype=np.float32,
        )

    model = _get_embedding_model()

    if model is None:

        return np.empty(
            (0, 0),
            dtype=np.float32,
        )

    try:

        all_embeddings = []

        total = len(texts)

        for index, text in enumerate(
            texts,
            start=1,
        ):

            print(
                f"[retriever] Embedding "
                f"{index}/{total}..."
            )

            result = list(
                model.embed([text])
            )

            if not result:
                continue

            vector = np.asarray(
                result[0],
                dtype=np.float32,
            )

            all_embeddings.append(
                vector
            )

        if not all_embeddings:

            return np.empty(
                (0, 0),
                dtype=np.float32,
            )

        return np.asarray(
            all_embeddings,
            dtype=np.float32,
        )

    except Exception as exc:

        print(
            "[retriever] Embedding generation failed: "
            f"{exc}"
        )

        return np.empty(
            (0, 0),
            dtype=np.float32,
        )


def _normalize_embeddings(embeddings):

    if embeddings.size == 0:
        return embeddings

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    norms = np.maximum(
        norms,
        1e-12,
    )

    return embeddings / norms


# ============================================================
# METADATA
# ============================================================

def _ensure_metadata():

    global _metadata

    if _metadata is not None:
        return

    with _load_lock:

        if _metadata is not None:
            return

        if not os.path.exists(
            METADATA_PATH
        ):

            print(
                "[retriever] RAG metadata not found. "
                "Building dataset index..."
            )

            try:

                from rag.build_index import build

                build()

            except Exception as exc:

                print(
                    "[retriever] Unable to build dataset "
                    f"metadata: {exc}"
                )

                _metadata = []

                return

        try:

            with open(
                METADATA_PATH,
                "r",
                encoding="utf-8",
            ) as handle:

                loaded = json.load(
                    handle
                )

            if isinstance(
                loaded,
                list,
            ):

                _metadata = loaded

            else:

                _metadata = []

                print(
                    "[retriever] Invalid metadata format."
                )

        except Exception as exc:

            print(
                "[retriever] Failed to load metadata: "
                f"{exc}"
            )

            _metadata = []


# ============================================================
# DATASET EMBEDDINGS
# ============================================================

def _load_dataset_embeddings():

    global _metadata_embeddings

    if _metadata_embeddings is not None:
        return

    _ensure_metadata()

    if not _metadata:

        _metadata_embeddings = np.empty(
            (0, 0),
            dtype=np.float32,
        )

        return

    with _load_lock:

        if _metadata_embeddings is not None:
            return

        if os.path.exists(
            EMBEDDINGS_PATH
        ):

            try:

                cached = np.load(
                    EMBEDDINGS_PATH,
                    allow_pickle=False,
                )

                if (
                    cached.ndim == 2
                    and cached.shape[0]
                    == len(_metadata)
                    and cached.shape[1] > 0
                ):

                    _metadata_embeddings = (
                        _normalize_embeddings(
                            cached.astype(
                                np.float32
                            )
                        )
                    )

                    print(
                        "[retriever] Loaded cached "
                        "semantic embeddings: "
                        f"{len(_metadata)} records."
                    )

                    return

            except Exception as exc:

                print(
                    "[retriever] Could not load "
                    f"cached embeddings: {exc}"
                )

        documents = [
            _embedding_text(record)
            for record in _metadata
        ]

        print(
            "[retriever] Creating semantic embeddings "
            f"for {len(documents)} dataset records..."
        )

        embeddings = _encode_texts(
            documents
        )

        if embeddings.size == 0:

            _metadata_embeddings = np.empty(
                (0, 0),
                dtype=np.float32,
            )

            return

        _metadata_embeddings = (
            _normalize_embeddings(
                embeddings
            )
        )

        try:

            os.makedirs(
                INDEX_DIR,
                exist_ok=True,
            )

            np.save(
                EMBEDDINGS_PATH,
                _metadata_embeddings,
            )

            print(
                "[retriever] Semantic embeddings cached at: "
                f"{EMBEDDINGS_PATH}"
            )

        except Exception as exc:

            print(
                "[retriever] Warning: could not save "
                f"embedding cache: {exc}"
            )


# ============================================================
# FAISS INDEX
# ============================================================

def _load_faiss_index():

    global _faiss_index

    if _faiss_index is not None:
        return _faiss_index

    with _load_lock:

        if _faiss_index is not None:
            return _faiss_index

        try:

            import faiss

        except Exception as exc:

            print(
                "[retriever] FAISS is unavailable: "
                f"{exc}"
            )

            return None

        _load_dataset_embeddings()

        if (
            _metadata_embeddings is None
            or _metadata_embeddings.size == 0
        ):

            return None

        # ----------------------------------------------------
        # Load existing FAISS index
        # ----------------------------------------------------

        if os.path.exists(
            FAISS_INDEX_PATH
        ):

            try:

                index = faiss.read_index(
                    FAISS_INDEX_PATH
                )

                if (
                    index.ntotal
                    == len(_metadata)
                    and index.d
                    == _metadata_embeddings.shape[1]
                ):

                    _faiss_index = index

                    print(
                        "[retriever] Loaded FAISS "
                        f"index: {index.ntotal} vectors."
                    )

                    return _faiss_index

                print(
                    "[retriever] Existing FAISS index "
                    "does not match metadata. Rebuilding..."
                )

            except Exception as exc:

                print(
                    "[retriever] Could not load FAISS "
                    f"index: {exc}"
                )

        # ----------------------------------------------------
        # Build FAISS index
        # ----------------------------------------------------

        dimension = (
            _metadata_embeddings.shape[1]
        )

        index = faiss.IndexFlatIP(
            dimension
        )

        index.add(
            _metadata_embeddings.astype(
                np.float32
            )
        )

        os.makedirs(
            INDEX_DIR,
            exist_ok=True,
        )

        faiss.write_index(
            index,
            FAISS_INDEX_PATH,
        )

        _faiss_index = index

        print(
            "[retriever] FAISS index created: "
            f"{index.ntotal} vectors, "
            f"dimension: {index.d}"
        )

        return _faiss_index


# ============================================================
# DATASET RETRIEVAL USING FAISS
# ============================================================

def retrieve_similar_bugs(
    query_text: str,
    top_k: int = 3,
):

    query = _safe_text(
        query_text
    )

    if not query:
        return []

    index = _load_faiss_index()

    if index is None:
        return []

    try:

        query_embedding = _encode_texts(
            [query]
        )

        if query_embedding.size == 0:
            return []

        query_embedding = (
            _normalize_embeddings(
                query_embedding
            )
        )

        requested_k = max(
            1,
            int(top_k),
        )

        result_count = min(
            requested_k,
            index.ntotal,
        )

        scores, indices = index.search(
            query_embedding.astype(
                np.float32
            ),
            result_count,
        )

        results = []

        for score, index_id in zip(
            scores[0],
            indices[0],
        ):

            if index_id < 0:
                continue

            score = float(score)

            if score <= 0.05:
                continue

            results.append(
                {
                    "record": _metadata[
                        int(index_id)
                    ],
                    "similarity": float(
                        np.clip(
                            score,
                            0.0,
                            1.0,
                        )
                    ),
                     "source_type": "dataset"
                }
            )

        return results

    except Exception as exc:

        print(
            "[retriever] FAISS dataset retrieval "
            f"failed: {exc}"
        )

        return []


# ============================================================
# GITHUB ARTIFACT CACHE
# ============================================================

def _artifact_cache_key(
    artifact: dict,
):

    return (
        _safe_text(
            artifact.get("source")
        ),
        _safe_text(
            artifact.get("artifact_type")
        ),
        str(
            artifact.get("number")
        ),
        _safe_text(
            artifact.get("url")
        ),
    )


def _get_github_embeddings(
    artifacts,
):

    if not artifacts:

        return (
            np.empty(
                (0, 0),
                dtype=np.float32,
            ),
            [],
        )

    texts_to_encode = []
    artifacts_to_encode = []
    embeddings_by_key = {}

    for artifact in artifacts:

        key = _artifact_cache_key(
            artifact
        )

        cached = (
            _github_embedding_cache.get(
                key
            )
        )

        if cached is not None:

            embeddings_by_key[
                key
            ] = cached

        else:

            texts_to_encode.append(
                _github_artifact_text(
                    artifact
                )
            )

            artifacts_to_encode.append(
                (
                    key,
                    artifact,
                )
            )

    if texts_to_encode:

        new_embeddings = _encode_texts(
            texts_to_encode
        )

        if new_embeddings.size != 0:

            new_embeddings = (
                _normalize_embeddings(
                    new_embeddings
                )
            )

            for index, (
                key,
                artifact,
            ) in enumerate(
                artifacts_to_encode
            ):

                vector = new_embeddings[
                    index
                ]

                _github_embedding_cache[
                    key
                ] = vector

                embeddings_by_key[
                    key
                ] = vector

    ordered_embeddings = []
    ordered_artifacts = []

    for artifact in artifacts:

        key = _artifact_cache_key(
            artifact
        )

        vector = embeddings_by_key.get(
            key
        )

        if vector is None:
            continue

        ordered_embeddings.append(
            vector
        )

        ordered_artifacts.append(
            artifact
        )

    if not ordered_embeddings:

        return (
            np.empty(
                (0, 0),
                dtype=np.float32,
            ),
            [],
        )

    return (
        np.asarray(
            ordered_embeddings,
            dtype=np.float32,
        ),
        ordered_artifacts,
    )


# ============================================================
# GITHUB ISSUE / PR RETRIEVAL
# ============================================================

def retrieve_similar_github_artifacts(
    query_text: str,
    artifacts,
    top_k: int = 3,
):

    query = _safe_text(
        query_text
    )

    if not query or not artifacts:
        return []

    try:

        query_embedding = _encode_texts(
            [query]
        )

        if query_embedding.size == 0:
            return []

        github_embeddings, valid_artifacts = (
            _get_github_embeddings(
                artifacts
            )
        )

        if github_embeddings.size == 0:
            return []

        query_embedding = (
            _normalize_embeddings(
                query_embedding
            )
        )

        github_embeddings = (
            _normalize_embeddings(
                github_embeddings
            )
        )

        scores = np.dot(
            github_embeddings,
            query_embedding[0],
        )

        requested_k = max(
            1,
            int(top_k),
        )

        result_count = min(
            requested_k,
            len(scores),
        )

        indices = np.argsort(
            scores
        )[::-1][:result_count]

        results = []

        for index in indices:

            score = float(
                scores[index]
            )

            if score <= 0.05:
                continue

            results.append(
                {
                    "record": valid_artifacts[
                        int(index)
                    ],
                    "similarity": float(
                        np.clip(
                            score,
                            0.0,
                            1.0,
                        )
                    ),
                    "source_type": "github"
                }
            )

        return results

    except Exception as exc:

        print(
            "[retriever] Semantic GitHub retrieval "
            f"failed: {exc}"
        )

        return []


# ============================================================
# FINDING QUERY
# ============================================================

def build_finding_query(
    finding: dict,
):

    parts = []

    fields = [
        ("file", "File"),
        ("bug_type", "Bug type"),
        ("rule", "Detection rule"),
        ("error", "Error"),
        ("cause", "Cause"),
        ("current_code", "Code"),
    ]

    for key, label in fields:

        value = _safe_text(
            finding.get(key)
        )

        if value:

            parts.append(
                f"{label}: {value}"
            )

    return "\n".join(parts)


# ============================================================
# CONVENIENCE RETRIEVAL
# ============================================================

def retrieve_for_finding(
    finding: dict,
    top_k: int = 3,
):

    query = build_finding_query(
        finding
    )

    return retrieve_similar_bugs(
        query_text=query,
        top_k=top_k,
    )


# ============================================================
# STATUS
# ============================================================

def get_retriever_status():

    _ensure_metadata()

    index = _load_faiss_index()

    return {
        "index_type": (
            "FAISS IndexFlatIP + "
            "FastEmbed Semantic Embeddings"
        ),
        "embedding_model": (
            EMBEDDING_MODEL_NAME
        ),
        "loaded": (
            index is not None
            and index.ntotal > 0
        ),
        "records": (
            len(_metadata)
            if _metadata
            else 0
        ),
        "embedding_dimension": (
            int(index.d)
            if index is not None
            else 0
        ),
        "faiss_vectors": (
            int(index.ntotal)
            if index is not None
            else 0
        ),
        "faiss_index_path": (
            FAISS_INDEX_PATH
        ),
        "faiss_exists": (
            os.path.exists(
                FAISS_INDEX_PATH
            )
        ),
    }