"""
Entry point for the backend.

Flow:
GitHub URL
    -> validate repository
    -> download repository
    -> scan source files
    -> detect bug candidates
    -> retrieve dataset bugs using RAG
    -> retrieve GitHub Issues / Pull Requests
    -> combine historical evidence
    -> LLM analysis
    -> confidence estimation
    -> final bug report
"""

import os
import uuid
import concurrent.futures
import threading

import numpy as np

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from models import (
    ScanRequest,
    ScanResult,
    ScanSummary,
    BugReport,
    RetrievedBug,
)

from github_handler import (
    validate_repo,
    download_repo,
    cleanup_repo,
    fetch_historical_artifacts,
)

from file_scanner import (
    find_source_files,
    read_file_safely,
)

from detectors.python_detector import (
    detect as detect_python,
)

from detectors.js_detector import (
    detect as detect_js,
)

from detectors.cfamily_detector import (
    detect as detect_cfamily,
)

from rag.retriever import (
    retrieve_similar_bugs,
)

from llm_client import (
    analyze_finding,
    get_fallback_report,
    analyze_file,
)


app = FastAPI(
    title="RAG-Enhanced LLM for GitHub Bug Detection and Recovery"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


DETECTORS_BY_EXTENSION = {
    ".py": detect_python,

    ".js": detect_js,
    ".jsx": detect_js,
    ".ts": detect_js,
    ".tsx": detect_js,

    ".java": detect_cfamily,
    ".c": detect_cfamily,
    ".cpp": detect_cfamily,
    ".cs": detect_cfamily,
    ".go": detect_cfamily,
    ".php": detect_cfamily,
}


MAX_LLM_CALLS_PER_SCAN = 8

MAX_PARALLEL_WORKERS = 6

MAX_FILE_SCAN_WORKERS = 12

MAX_GITHUB_ISSUES = 30

MAX_GITHUB_PULL_REQUESTS = 30

MAX_GITHUB_RAG_RESULTS = 3


# -------------------------------------------------------------------
# GitHub semantic model cache
# -------------------------------------------------------------------

_github_embedding_model = None

_github_embedding_lock = threading.Lock()


def _get_github_embedding_model():
    """
    Load the same Sentence Transformer model used by the
    main RAG index.

    The model is loaded only once per backend process.
    """

    global _github_embedding_model

    if _github_embedding_model is not None:
        return _github_embedding_model

    with _github_embedding_lock:

        if _github_embedding_model is not None:
            return _github_embedding_model

        try:

            from sentence_transformers import (
                SentenceTransformer,
            )

            print(
                "[github-rag] Loading embedding model..."
            )

            _github_embedding_model = (
                SentenceTransformer(
                    "all-MiniLM-L6-v2"
                )
            )

            print(
                "[github-rag] Embedding model READY."
            )

        except Exception as e:

            print(
                "[github-rag] Embedding model unavailable: "
                f"{e}"
            )

            _github_embedding_model = None

    return _github_embedding_model


# -------------------------------------------------------------------
# Basic helpers
# -------------------------------------------------------------------

def _empty_summary(
    repo: str,
    message: str,
) -> ScanResult:

    return ScanResult(
        summary=ScanSummary(
            repo=repo,
            files_scanned=0,
            bugs_found=0,
            confidence=100,
            error_level="Less Errors",
            scan_status=message,
        ),
        bugs=[],
    )


def _compute_error_level(
    bug_reports,
):

    issue_count = len(
        bug_reports
    )

    if issue_count <= 10:
        return "Less Errors"

    if issue_count <= 30:
        return "Medium Errors"

    return "More Errors"


def _clamp_confidence(
    value,
) -> int:

    try:

        number = int(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return 70

    return max(
        0,
        min(
            100,
            number,
        ),
    )


def _compute_confidence(
    bug_reports,
):

    if not bug_reports:
        return 100

    return round(
        sum(
            b.confidence
            for b in bug_reports
        )
        / len(bug_reports)
    )


# -------------------------------------------------------------------
# GitHub historical artifact preparation
# -------------------------------------------------------------------

def _github_artifact_to_record(
    artifact: dict,
):
    """
    Convert a GitHub Issue / Pull Request into the same
    general record structure expected by the RAG/LLM layer.
    """

    title = str(
        artifact.get(
            "title"
        )
        or ""
    ).strip()

    body = str(
        artifact.get(
            "body"
        )
        or ""
    ).strip()

    patch = str(
        artifact.get(
            "patch"
        )
        or ""
    ).strip()

    artifact_type = artifact.get(
        "artifact_type",
        "github",
    )

    labels = artifact.get(
        "labels",
        [],
    )

    if not isinstance(
        labels,
        list,
    ):

        labels = []

    label_text = ", ".join(
        str(label)
        for label in labels
        if label
    )

    if artifact_type == "pull_request":

        description = (
            f"Historical GitHub Pull Request: "
            f"{title}"
        )

    else:

        description = (
            f"Historical GitHub Issue: "
            f"{title}"
        )

    if body:

        description += (
            f"\n{body}"
        )

    if label_text:

        description += (
            f"\nLabels: {label_text}"
        )

    solution_parts = []

    if body:
        solution_parts.append(
            body
        )

    if patch:
        solution_parts.append(
            patch
        )

    solution = "\n".join(
        solution_parts
    )

    return {
        "dataset_source": (
            "GitHub Pull Request"
            if artifact_type == "pull_request"
            else "GitHub Issue"
        ),

        "bug_type": (
            label_text
            if label_text
            else "Historical GitHub Bug"
        ),

        "bug_description": description,

        "solution": solution,

        "error": title,

        "language": "unknown",

        "buggy_code": patch,

        "github_url": artifact.get(
            "url"
        ),

        "github_number": artifact.get(
            "number"
        ),

        "artifact_type": artifact_type,

        "title": title,

        "body": body,

        "patch": patch,
    }


def _retrieve_github_history(
    query_text: str,
    artifacts: list,
):
    """
    Perform semantic retrieval over the repository's
    historical GitHub Issues and Pull Requests.

    Returns records in the same structure used by
    retrieve_similar_bugs().
    """

    if not artifacts:
        return []

    model = _get_github_embedding_model()

    if model is None:

        print(
            "[github-rag] Semantic model unavailable. "
            "Skipping GitHub semantic retrieval."
        )

        return []

    records = []

    texts = []

    for artifact in artifacts:

        record = _github_artifact_to_record(
            artifact
        )

        text_parts = [
            record.get(
                "bug_description",
                ""
            ),
            record.get(
                "error",
                ""
            ),
            record.get(
                "bug_type",
                ""
            ),
            record.get(
                "buggy_code",
                ""
            ),
            record.get(
                "solution",
                ""
            ),
        ]

        text = "\n".join(
            str(part)
            for part in text_parts
            if part
        ).strip()

        if not text:
            continue

        records.append(
            record
        )

        texts.append(
            text
        )

    if not texts:
        return []

    try:

        embeddings = model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        history_embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        query_vector = np.asarray(
            embeddings[0],
            dtype="float32",
        )

        history_matrix = np.asarray(
            history_embeddings,
            dtype="float32",
        )

        scores = np.dot(
            history_matrix,
            query_vector,
        )

        ranked_indices = np.argsort(
            scores
        )[::-1]

        results = []

        for index in ranked_indices[
            :MAX_GITHUB_RAG_RESULTS
        ]:

            similarity = float(
                scores[index]
            )

            similarity = max(
                0.0,
                min(
                    1.0,
                    similarity,
                ),
            )

            if similarity < 0.10:
                continue

            results.append(
                {
                    "record": records[
                        int(index)
                    ],
                    "similarity": similarity,
                }
            )

        print(
            "[github-rag] Retrieved "
            f"{len(results)} relevant historical artifacts."
        )

        return results

    except Exception as e:

        print(
            "[github-rag] Semantic retrieval failed: "
            f"{e}"
        )

        return []


# -------------------------------------------------------------------
# API endpoints
# -------------------------------------------------------------------

@app.get(
    "/api/health"
)
def health():

    return {
        "status": "ok"
    }


@app.post(
    "/api/validate"
)
def validate_repository(
    req: ScanRequest,
):

    try:

        info = validate_repo(
            req.repo_url
        )

        return {
            "status": "valid",
            "message": "Repository is valid.",
            "owner": info["owner"],
            "name": info["repo"],
            "default_branch": info[
                "default_branch"
            ],
        }

    except Exception as e:

        return {
            "status": "invalid",
            "message": str(e),
        }


# -------------------------------------------------------------------
# Main scan
# -------------------------------------------------------------------

@app.post(
    "/api/scan",
    response_model=ScanResult,
)
def scan_repo(
    req: ScanRequest,
):

    try:

        repository = validate_repo(
            req.repo_url
        )

    except Exception as e:

        return _empty_summary(
            req.repo_url,
            str(e),
        )

    owner = repository[
        "owner"
    ]

    repo_name = repository[
        "repo"
    ]

    default_branch = repository[
        "default_branch"
    ]

    repo_path = None

    temp_root = None

    try:

        # ------------------------------------------------------------
        # Download repository
        # ------------------------------------------------------------

        repo_path, temp_root = download_repo(
            req.repo_url
        )

        # ------------------------------------------------------------
        # Find source files
        # ------------------------------------------------------------

        files = find_source_files(
            repo_path
        )

        # ------------------------------------------------------------
        # Historical GitHub Issues + Pull Requests
        # ------------------------------------------------------------

        github_history = {
            "issues": [],
            "pull_requests": [],
            "artifacts": [],
        }

        try:

            github_history = (
                fetch_historical_artifacts(
                    req.repo_url,
                    max_issues=MAX_GITHUB_ISSUES,
                    max_pull_requests=MAX_GITHUB_PULL_REQUESTS,
                )
            )

        except Exception as e:

            print(
                "[scan] Historical GitHub retrieval "
                f"failed: {e}"
            )

        github_artifacts = github_history.get(
            "artifacts",
            [],
        )

        print(
            "[scan] Historical GitHub artifacts: "
            f"{len(github_artifacts)}"
        )

        # ------------------------------------------------------------
        # Stage 1: local file detection
        # ------------------------------------------------------------

        def _scan_one_file(
            full_path,
        ):

            ext = os.path.splitext(
                full_path
            )[1].lower()

            source = read_file_safely(
                full_path
            )

            if not source:
                return []

            relative_path = os.path.relpath(
                full_path,
                repo_path,
            )

            try:

                detector = (
                    DETECTORS_BY_EXTENSION.get(
                        ext
                    )
                )

                if detector:

                    findings = detector(
                        relative_path,
                        source,
                    )

                else:

                    findings = analyze_file(
                        relative_path,
                        source,
                    )

            except Exception as e:

                print(
                    "[scan] Failed to analyze "
                    f"{relative_path}: {e}"
                )

                return []

            return [
                (
                    finding,
                    relative_path,
                )
                for finding in findings
            ]

        file_findings = [
            None
        ] * len(files)

        if files:

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=MAX_FILE_SCAN_WORKERS
            ) as pool:

                future_to_index = {
                    pool.submit(
                        _scan_one_file,
                        path,
                    ): index

                    for index, path
                    in enumerate(files)
                }

                for future in (
                    concurrent.futures.as_completed(
                        future_to_index
                    )
                ):

                    index = (
                        future_to_index[
                            future
                        ]
                    )

                    file_findings[
                        index
                    ] = future.result()

        pending = [
            item
            for findings in file_findings
            if findings
            for item in findings
        ]

        # ------------------------------------------------------------
        # Stage 2: RAG + GitHub history + LLM
        # ------------------------------------------------------------

        def _analyze_one(
            index,
            finding,
            relative_path,
        ):

            query_text = (
                f"{finding.get('error', '')}\n"
                f"{finding.get('bug_type', '')}\n"
                f"{finding.get('current_code', '')}"
            )

            # Dataset RAG retrieval
            dataset_results = (
                retrieve_similar_bugs(
                    query_text=query_text,
                    top_k=3,
                )
            )

            # GitHub Issues / PR semantic retrieval
            github_results = (
                _retrieve_github_history(
                    query_text,
                    github_artifacts,
                )
            )

            # Combine both historical evidence sources.
            retrieved = (
                dataset_results
                + github_results
            )

            retrieved.sort(
                key=lambda item: item.get(
                    "similarity",
                    0,
                ),
                reverse=True,
            )

            retrieved = retrieved[
                :5
            ]

            if index < MAX_LLM_CALLS_PER_SCAN:

                analysis = analyze_finding(
                    finding,
                    retrieved,
                )

            else:

                analysis = get_fallback_report(
                    finding
                )

            return (
                finding,
                relative_path,
                retrieved,
                analysis,
            )

        results = [
            None
        ] * len(pending)

        if pending:

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=MAX_PARALLEL_WORKERS
            ) as pool:

                future_to_index = {
                    pool.submit(
                        _analyze_one,
                        index,
                        finding,
                        relative_path,
                    ): index

                    for index, (
                        finding,
                        relative_path,
                    )
                    in enumerate(pending)
                }

                for future in (
                    concurrent.futures.as_completed(
                        future_to_index
                    )
                ):

                    index = (
                        future_to_index[
                            future
                        ]
                    )

                    try:

                        results[
                            index
                        ] = future.result()

                    except Exception as e:

                        print(
                            "[scan] Finding analysis "
                            f"failed: {e}"
                        )

                        results[
                            index
                        ] = None

        # ------------------------------------------------------------
        # Build bug reports
        # ------------------------------------------------------------

        bug_reports = []

        bug_number = 0

        ai_notice = None

        for result in results:

            if result is None:
                continue

            (
                finding,
                relative_path,
                retrieved,
                analysis,
            ) = result

            if (
                ai_notice is None
                and analysis.get(
                    "rate_limited"
                )
            ):

                ai_notice = analysis.get(
                    "rate_limit_message"
                )

            bug_number += 1

            reported_bug_type = (
                analysis.get(
                    "bug_type",
                    finding.get(
                        "bug_type",
                        "Other",
                    ),
                )
            )

            if finding.get(
                "rule"
            ) == "unreachable_code":

                reported_bug_type = (
                    "Unreachable Code"
                )

            retrieved_bugs = []

            for item in retrieved:

                record = item.get(
                    "record",
                    {},
                )

                retrieved_bugs.append(
                    RetrievedBug(
                        dataset_source=(
                            record.get(
                                "dataset_source"
                            )
                            or record.get(
                                "source"
                            )
                            or "Unknown"
                        ),

                        bug_type=record.get(
                            "bug_type"
                        ),

                        bug_description=record.get(
                            "bug_description"
                        )
                        or record.get(
                            "title"
                        ),

                        solution=record.get(
                            "solution"
                        ),

                        similarity=round(
                            float(
                                item.get(
                                    "similarity",
                                    0,
                                )
                            )
                            * 100,
                            1,
                        ),
                    )
                )

            bug_reports.append(
                BugReport(
                    id=str(
                        uuid.uuid4()
                    )[:8],

                    number=bug_number,

                    error=analysis.get(
                        "error",
                        finding.get(
                            "error",
                            "Possible issue",
                        ),
                    ),

                    bug_type=reported_bug_type,

                    file=relative_path,

                    function=finding.get(
                        "function"
                    ),

                    line_start=finding.get(
                        "line_start"
                    ),

                    line_end=finding.get(
                        "line_end"
                    ),

                    line_note=(
                        None
                        if finding.get(
                            "line_start"
                        )
                        else
                        "Exact line could not be determined."
                    ),

                    cause=analysis.get(
                        "cause",
                        finding.get(
                            "cause",
                            "",
                        ),
                    ),

                    why_occurs=analysis.get(
                        "why_occurs"
                    ),

                    solution_type=analysis.get(
                        "solution_type",
                        "replace",
                    ),

                    solution=analysis.get(
                        "solution",
                        "",
                    ),

                    current_code=finding.get(
                        "current_code",
                        "",
                    ),

                    replacement_code=analysis.get(
                        "replacement_code"
                    ),

                    add_location=analysis.get(
                        "add_location"
                    ),

                    new_file_path=analysis.get(
                        "new_file_path"
                    ),

                    explanation=analysis.get(
                        "explanation"
                    ),

                    confidence=_clamp_confidence(
                        analysis.get(
                            "confidence"
                        )
                    ),

                    retrieved_bugs=retrieved_bugs,

                    insufficient_evidence=bool(
                        analysis.get(
                            "insufficient_evidence",
                            False,
                        )
                    ),
                )
            )

        # ------------------------------------------------------------
        # Final summary
        # ------------------------------------------------------------

        confidence = _compute_confidence(
            bug_reports
        )

        error_level = _compute_error_level(
            bug_reports
        )

        return ScanResult(
            summary=ScanSummary(
                repo=f"{owner}/{repo_name}",

                files_scanned=len(
                    files
                ),

                bugs_found=len(
                    bug_reports
                ),

                confidence=confidence,

                error_level=error_level,

                scan_status="Completed",

                ai_notice=ai_notice,
            ),

            bugs=bug_reports,
        )

    except Exception as e:

        print(
            f"[scan] Scan failed: {e}"
        )

        return _empty_summary(
            req.repo_url,
            (
                "Scan failed. "
                "Please try again later."
            ),
        )

    finally:

        if temp_root:

            cleanup_repo(
                temp_root
            )

        elif repo_path:

            cleanup_repo(
                repo_path
            )


# -------------------------------------------------------------------
# Frontend
# -------------------------------------------------------------------

frontend_dir = os.path.join(
    os.path.dirname(__file__),
    "..",
    "frontend",
)


if os.path.isdir(
    frontend_dir
):

    app.mount(
        "/",
        StaticFiles(
            directory=frontend_dir,
            html=True,
        ),
        name="frontend",
    )
