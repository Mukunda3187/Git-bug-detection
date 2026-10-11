"""
Entry point for the backend.
Architecture:
GitHub Repository
        ↓
Source File Extraction
        ↓
Parse & Chunk Code
        ↓
Local Bug Detection
        ↓
Create Bug Query
        ↓
Semantic Embedding + FAISS RAG
        ↓
Historical GitHub Issues + PRs
        ↓
LLM Analysis
        ↓
Confidence Check
        ↓
Final Bug Report
"""
import os
import uuid
import concurrent.futures
import threading
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
    check_repository,
    download_repository,
    cleanup,
    fetch_historical_artifacts,
)
from file_scanner import (
    find_source_files,
    read_file_safely,
)
from detectors.python_detector import detect as detect_python
from detectors.js_detector import detect as detect_js
from detectors.cfamily_detector import detect as detect_cfamily
from code_chunker import chunk_source_file
from code_validator import apply_validation_to_result
from rag.retriever import (
    retrieve_similar_bugs,
    retrieve_similar_github_artifacts,
)
from llm_client import (
    analyze_finding,
    get_llm_unavailable_report,
    LLM_UNAVAILABLE_MESSAGE,
    analyze_file,
)
# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(
    title="RAG-Enhanced LLM for GitHub Bug Detection and Recovery"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# ============================================================
# LANGUAGE DETECTORS
# ============================================================
DETECTORS_BY_EXTENSION = {
    # Python
    ".py": detect_python,
    ".pyi": detect_python,
    # JavaScript / TypeScript
    ".js": detect_js,
    ".jsx": detect_js,
    ".ts": detect_js,
    ".tsx": detect_js,
    # C-family / structural detectors
    ".java": detect_cfamily,
    ".c": detect_cfamily,
    ".cpp": detect_cfamily,
    ".cs": detect_cfamily,
    ".go": detect_cfamily,
    ".php": detect_cfamily,
}
# ============================================================
# SCAN SETTINGS
# ============================================================
# Maximum number of findings that receive real LLM analysis.
MAX_LLM_CALLS_PER_SCAN = None
# Number of concurrent local file scanning workers.
MAX_FILE_SCAN_WORKERS = 12
# ============================================================
# REAL SCAN PROGRESS
# ============================================================

SCAN_JOBS = {}
SCAN_JOBS_LOCK = threading.Lock()
# Allow up to 4 concurrent scans so a new request never waits
# for a previous scan that is still running or hung.
SCAN_JOB_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=4)


def _model_to_dict(value):
    """Support both Pydantic v1 and v2."""
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value.dict()


def _new_scan_job(scan_id, repo_url):
    with SCAN_JOBS_LOCK:
        SCAN_JOBS[scan_id] = {
            "scan_id": scan_id,
            "repo_url": repo_url,
            "status": "queued",
            "phase": "Waiting to start",
            "progress": 0,
            "files_processed": 0,
            "files_total": 0,
            "current_file": "",
            "findings_processed": 0,
            "findings_total": 0,
            "result": None,
            "error": None,
            # Progressive bug streaming: bugs are appended here
            # one by one as the LLM finishes each analysis.
            "completed_bugs": [],
        }


def _update_scan_progress(
    scan_id,
    *,
    status=None,
    phase=None,
    progress=None,
    files_processed=None,
    files_total=None,
    current_file=None,
    findings_processed=None,
    findings_total=None,
):
    if not scan_id:
        return

    with SCAN_JOBS_LOCK:
        job = SCAN_JOBS.get(scan_id)
        if not job:
            return

        if status is not None:
            job["status"] = status
        if phase is not None:
            job["phase"] = phase
        if progress is not None:
            job["progress"] = max(0, min(100, int(progress)))
        if files_processed is not None:
            job["files_processed"] = int(files_processed)
        if files_total is not None:
            job["files_total"] = int(files_total)
        if current_file is not None:
            job["current_file"] = current_file
        if findings_processed is not None:
            job["findings_processed"] = int(findings_processed)
        if findings_total is not None:
            job["findings_total"] = int(findings_total)


def _get_scan_job(scan_id):
    with SCAN_JOBS_LOCK:
        job = SCAN_JOBS.get(scan_id)
        return dict(job) if job else None


def _push_completed_bug(scan_id, bug_dict):
    """Append a finished BugReport dict to the job's streaming list."""
    if not scan_id:
        return
    with SCAN_JOBS_LOCK:
        job = SCAN_JOBS.get(scan_id)
        if job is not None:
            job["completed_bugs"].append(bug_dict)


def _run_scan_job(scan_id, req):
    # Immediately mark as running so the frontend stops showing "Waiting to start"
    _update_scan_progress(
        scan_id,
        status="running",
        phase="Validating repository",
        progress=2,
    )
    try:

        result = _run_scan(req, scan_id=scan_id)

        with SCAN_JOBS_LOCK:
            job = SCAN_JOBS.get(scan_id)
            if job:
                job["status"] = "completed"
                job["phase"] = "Completed"
                job["progress"] = 100
                job["current_file"] = ""
                job["result"] = _model_to_dict(result)

    except Exception as exc:
        print(f"[scan] Background scan failed: {exc}")

        with SCAN_JOBS_LOCK:
            job = SCAN_JOBS.get(scan_id)
            if job:
                job["status"] = "failed"
                job["phase"] = "Scan failed"
                job["error"] = str(exc)
                job["progress"] = 100


# ============================================================
# EMPTY RESPONSE
# ============================================================
def _empty_summary(repo: str, message: str) -> ScanResult:
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
# ============================================================
# ERROR LEVEL
# ============================================================
def _compute_error_level(bug_reports):
    issue_count = len(bug_reports)
    if issue_count <= 10:
        return "Less Errors"
    elif issue_count <= 30:
        return "Medium Errors"
    else:
        return "More Errors"
# ============================================================
# CONFIDENCE CLAMP
# ============================================================
def _clamp_confidence(value) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 70
    return max(
        0,
        min(100, number)
    )
# ============================================================
# OVERALL CONFIDENCE
# ============================================================
def _compute_confidence(bug_reports):
    if not bug_reports:
        return 100
    return round(
        sum(
            bug.confidence
            for bug in bug_reports
        )
        / len(bug_reports)
    )
# ============================================================
# FIND BEST CHUNK FOR A FINDING
# ============================================================
def _find_best_chunk(
    chunks,
    line_start,
    line_end,
):
    """
    Find the code chunk that overlaps most with
    the detected bug location.
    """
    if not chunks:
        return None
    try:
        finding_start = int(line_start)
    except (TypeError, ValueError):
        finding_start = None
    try:
        finding_end = int(line_end)
    except (TypeError, ValueError):
        finding_end = finding_start
    # If detector did not provide line information,
    # use the first chunk as a safe fallback.
    if finding_start is None:
        return chunks[0]
    if finding_end is None:
        finding_end = finding_start
    best_chunk = None
    best_overlap = -1
    for chunk in chunks:
        chunk_start = chunk.get(
            "line_start",
            0,
        )
        chunk_end = chunk.get(
            "line_end",
            0,
        )
        overlap_start = max(
            finding_start,
            chunk_start,
        )
        overlap_end = min(
            finding_end,
            chunk_end,
        )
        if overlap_start <= overlap_end:
            overlap = (
                overlap_end
                - overlap_start
                + 1
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_chunk = chunk
    # If no direct overlap was found,
    # choose the closest chunk.
    if best_chunk is None:
        best_distance = None
        for chunk in chunks:
            chunk_start = chunk.get(
                "line_start",
                0,
            )
            chunk_end = chunk.get(
                "line_end",
                0,
            )
            if finding_end < chunk_start:
                distance = (
                    chunk_start
                    - finding_end
                )
            elif finding_start > chunk_end:
                distance = (
                    finding_start
                    - chunk_end
                )
            else:
                distance = 0
            if (
                best_distance is None
                or distance < best_distance
            ):
                best_distance = distance
                best_chunk = chunk
    return best_chunk
# ============================================================
# HEALTH
# ============================================================
@app.get("/api/health")
def health():
    return {
        "status": "ok"
    }
# ============================================================
# VALIDATE REPOSITORY
# ============================================================
@app.post("/api/validate")
def validate_repo(req: ScanRequest):
    return check_repository(
        req.repo_url
    )
# ============================================================
# MAIN SCAN
# ============================================================
def _run_scan(
    req: ScanRequest,
    scan_id=None,
):
    # ========================================================
    # 1. VALIDATE GITHUB REPOSITORY
    # ========================================================
    status = check_repository(
        req.repo_url
    )
    if status.status != "valid":
        return _empty_summary(
            req.repo_url,
            status.message,
        )

    _update_scan_progress(
        scan_id,
        phase="Repository validated",
        progress=8,
    )

    repo_path = None
    # ========================================================
    # 2. DOWNLOAD REPOSITORY
    # ========================================================
    try:
        repo_path = download_repository(
            status.owner,
            status.name,
            status.default_branch,
        )
    except Exception as exc:
        return _empty_summary(
            req.repo_url,
            f"Unable to download repository: {exc}",
        )

    _update_scan_progress(
        scan_id,
        phase="Repository downloaded",
        progress=15,
    )

    try:
        # ====================================================
        # 3. HISTORICAL GITHUB ISSUES + PRs
        # ====================================================

        _update_scan_progress(
            scan_id,
            phase="Retrieving historical GitHub issues and PRs",
            progress=18,
        )

        historical_artifacts = []
        if req.rag_enabled:
            try:
                historical = fetch_historical_artifacts(
                    req.repo_url,
                    max_issues=30,
                    max_pull_requests=30,
                )
                historical_artifacts = historical.get(
                    "artifacts",
                    [],
                )
                issue_count = len(
                    historical.get(
                        "issues",
                        [],
                    )
                )
                pull_request_count = len(
                    historical.get(
                        "pull_requests",
                        [],
                    )
                )
                print(
                    "[github] Historical artifacts retrieved: "
                    f"{len(historical_artifacts)} "
                    f"(Issues: {issue_count}, "
                    f"PRs: {pull_request_count})"
                )
            except Exception as exc:
                print(
                    "[github] Historical artifact retrieval failed: "
                    f"{exc}"
                )
                historical_artifacts = []
        else:
            print("[rag] Disabled: skipping historical GitHub retrieval.")

        _update_scan_progress(
            scan_id,
            phase="Finding source files",
            progress=25,
        )

        # ====================================================
        # 4. FIND SOURCE FILES
        # ====================================================
        files = find_source_files(
            repo_path
        )
        print(
            f"[scan] Files discovered: {len(files)}"
        )

        _update_scan_progress(
            scan_id,
            status="running",
            phase="Scanning files",
            progress=25 if files else 70,
            files_processed=0,
            files_total=len(files),
            current_file="",
        )

        # ====================================================
        # 5. PARSE + CHUNK + DETECT BUGS
        # ====================================================
        def _scan_one_file(full_path):
            extension = os.path.splitext(
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
                # ------------------------------------------------
                # PARSE & CHUNK CODE
                # ------------------------------------------------
                chunks = chunk_source_file(
                    relative_path,
                    source,
                )
                print(
                    f"[chunk] {relative_path}: "
                    f"{len(chunks)} chunks"
                )
                # ------------------------------------------------
                # BUG DETECTOR
                # ------------------------------------------------
                detector = DETECTORS_BY_EXTENSION.get(
                    extension
                )
                if detector:
                    findings = detector(
                        relative_path,
                        source,
                    )
                else:
                    # For languages without a dedicated
                    # detector, allow the LLM file analyzer
                    # to inspect the source.
                    findings = analyze_file(
                        relative_path,
                        source,
                    )
                if not findings:
                    return []
                # ------------------------------------------------
                # ATTACH RELEVANT CODE CHUNK
                # ------------------------------------------------
                for finding in findings:
                    best_chunk = _find_best_chunk(
                        chunks,
                        finding.get(
                            "line_start"
                        ),
                        finding.get(
                            "line_end"
                        ),
                    )
                    if best_chunk:
                        finding["code_chunk"] = (
                            best_chunk.get(
                                "code",
                                "",
                            )
                        )
                        finding["chunk_name"] = (
                            best_chunk.get(
                                "name"
                            )
                        )
                        finding["chunk_type"] = (
                            best_chunk.get(
                                "chunk_type"
                            )
                        )
                        finding["chunk_line_start"] = (
                            best_chunk.get(
                                "line_start"
                            )
                        )
                        finding["chunk_line_end"] = (
                            best_chunk.get(
                                "line_end"
                            )
                        )
                        finding["chunk_context"] = (
                            f"{best_chunk.get('chunk_type', 'code')} "
                            f"{best_chunk.get('name', '')}\n"
                            f"Lines "
                            f"{best_chunk.get('line_start', '')}-"
                            f"{best_chunk.get('line_end', '')}\n"
                            f"{best_chunk.get('code', '')}"
                        )
                        if not finding.get("current_code"):
                            finding["current_code"] = best_chunk.get("code", "")
                    else:
                        finding["code_chunk"] = ""
                        finding["chunk_context"] = ""
                return [
                    (
                        finding,
                        relative_path,
                    )
                    for finding in findings
                ]
            except Exception as exc:
                print(
                    f"[scan] Failed to analyze "
                    f"{relative_path}: {exc}"
                )
                return []
        # ====================================================
        # PARALLEL FILE SCANNING
        # ====================================================
        file_findings = [
            None
        ] * len(files)

        files_processed = 0
        files_total = len(files)

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

                for future in concurrent.futures.as_completed(
                    future_to_index
                ):
                    index = future_to_index[
                        future
                    ]

                    try:
                        file_findings[index] = (
                            future.result()
                        )
                    except Exception as exc:
                        print(
                            f"[scan] File worker failed: "
                            f"{exc}"
                        )
                        file_findings[index] = []

                    files_processed += 1

                    current_relative = os.path.relpath(
                        files[index],
                        repo_path,
                    )

                    file_progress = (
                        25
                        + int(
                            (
                                files_processed
                                / files_total
                            ) * 45
                        )
                        if files_total
                        else 70
                    )

                    _update_scan_progress(
                        scan_id,
                        phase="Scanning files",
                        progress=file_progress,
                        files_processed=files_processed,
                        files_total=files_total,
                        current_file=current_relative,
                    )

        pending = [
            item
            for findings in file_findings
            if findings
            for item in findings
        ]
        print(
            f"[scan] Findings detected: "
            f"{len(pending)}"
        )
        print(
            "[scan] Historical GitHub artifacts available: "
            f"{len(historical_artifacts)}"
        )
        # ====================================================
        # 6. RAG + LLM ANALYSIS
        # ====================================================

        _update_scan_progress(
            scan_id,
            phase="Preparing RAG and LLM analysis",
            progress=70 if pending else 95,
            files_processed=files_total,
            files_total=files_total,
            current_file="",
            findings_processed=0,
            findings_total=len(pending),
        )

        def _analyze_one(finding, relative_path):
            # ------------------------------------------------
            # CREATE BUG QUERY
            # ------------------------------------------------
            query_text = (
                f"{finding.get('error', '')}\n"
                f"{finding.get('cause', '')}\n"
                f"{finding.get('current_code', '')}\n"
                f"{finding.get('bug_type', '')}\n"
                f"{finding.get('chunk_context', '')}"
            )
            # ------------------------------------------------
            # RETRIEVE HISTORICAL CONTEXT ONLY WHEN RAG IS ENABLED
            # ------------------------------------------------
            print(f"[rag] RAG enabled: {req.rag_enabled}")
            if req.rag_enabled:
                try:
                    retrieved_dataset = retrieve_similar_bugs(
                        query_text=query_text,
                        top_k=3,
                    )
                    print(f"[rag] Dataset retrieval returned {len(retrieved_dataset)} results")
                except Exception as exc:
                    print(f"[rag] Dataset retrieval failed: {exc}")
                    retrieved_dataset = []

                try:
                    retrieved_github = retrieve_similar_github_artifacts(
                        query_text=query_text,
                        artifacts=historical_artifacts,
                        top_k=3,
                    )
                    print(f"[rag] GitHub retrieval returned {len(retrieved_github)} results")
                except Exception as exc:
                    print(f"[rag] GitHub artifact retrieval failed: {exc}")
                    retrieved_github = []

                retrieved = retrieved_dataset + retrieved_github
            else:
                retrieved = []
            # ------------------------------------------------
            # SORT BY SIMILARITY
            # ------------------------------------------------
            retrieved = sorted(
                retrieved,
                key=lambda item: item.get(
                    "similarity",
                    0,
                ),
                reverse=True,
            )
            # Keep strongest six.
            retrieved = retrieved[:6]
            print(f"[rag] Final retrieved count for finding in {relative_path}: {len(retrieved)}")
            # ------------------------------------------------
            # LLM ANALYSIS
            # ------------------------------------------------
            try:
                # Every finding goes through the complete LLM analysis path.
                # Provider failures are reported without inventing explanations.
                analysis = analyze_finding(
                    finding,
                    retrieved,
                )
            except Exception as exc:
                print(
                    f"[llm] Analysis failed for "
                    f"{relative_path}: {exc}"
                )
                analysis = get_llm_unavailable_report(
                    finding,
                    error=exc,
                )
            # ------------------------------------------------
            # CONFIDENCE CHECK + CODE VALIDATION
            # ------------------------------------------------
            # Validate replacement code only when the LLM provided
            # a complete per-finding explanation.
            # This keeps the architecture flow explicit:
            # LLM Analysis -> Confidence Check -> Final Report.
            if not analysis.get("llm_unavailable"):
                try:
                    analysis = apply_validation_to_result(
                        analysis,
                        finding,
                    )
                except Exception as exc:
                    print(
                        f"[validator] Validation failed for "
                        f"{relative_path}: {exc}"
                    )
            return (
                finding,
                relative_path,
                retrieved,
                analysis,
            )
        # Analyze findings serially so the next LLM request starts only
        # after the current finding has a complete result.
        results = []
        if pending:
            for findings_processed, (
                finding,
                relative_path,
            ) in enumerate(pending, start=1):
                try:
                    results.append(
                        _analyze_one(finding, relative_path)
                    )
                except Exception as exc:
                    print(
                        "[analysis] Finding analysis failed: "
                        f"{exc}"
                    )
                    results.append(
                        (
                            finding,
                            relative_path,
                            [],
                            get_llm_unavailable_report(
                                finding,
                                error=exc,
                            ),
                        )
                    )

                analysis_progress = (
                    70
                    + int(
                        (findings_processed / len(pending)) * 25
                    )
                )

                _update_scan_progress(
                    scan_id,
                    phase="Analyzing findings",
                    progress=analysis_progress,
                    files_processed=files_total,
                    files_total=files_total,
                    current_file=relative_path,
                    findings_processed=findings_processed,
                    findings_total=len(pending),
                )

        _update_scan_progress(
            scan_id,
            phase="Building final bug report",
            progress=97,
            files_processed=files_total,
            files_total=files_total,
            current_file="",
            findings_processed=len(pending),
            findings_total=len(pending),
        )

        # 7. BUILD BUG REPORTS
        # ====================================================
        bug_reports = []
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
            # ------------------------------------------------
            # LLM availability notice
            # ------------------------------------------------
            if (
                ai_notice is None
                and analysis.get("llm_unavailable")
            ):
                ai_notice = LLM_UNAVAILABLE_MESSAGE
            # ------------------------------------------------
            # CONFIDENCE
            # ------------------------------------------------
            confidence = _clamp_confidence(
                analysis.get(
                    "confidence"
                )
            )
            confidence_level = analysis.get(
                "confidence_level"
            )
            if confidence_level == "High Confidence":
                confidence_status = (
                    "Potential Bug"
                )
            else:
                confidence_level = (
                    "Low Confidence"
                )
                confidence_status = (
                    "Uncertain Finding"
                )
            # ------------------------------------------------
            # BUG TYPE
            # ------------------------------------------------
            bug_type = analysis.get(
                "bug_type",
                finding.get(
                    "bug_type",
                    "Other",
                ),
            )
            if finding.get(
                "rule"
            ) == "unreachable_code":
                bug_type = "Unreachable Code"
            # ------------------------------------------------
            # CONVERT RETRIEVED EVIDENCE
            # ------------------------------------------------
            retrieved_bugs = []
            print(f"[rag] Converting {len(retrieved)} retrieved items to RetrievedBug objects...")
            for item in retrieved:
                record = item.get(
                    "record",
                    {},
                )
                source_type = item.get(
                    "source_type",
                    "dataset",
                )
                # ============================================
                # DATASET RESULT
                # ============================================
                if source_type == "dataset":
                    retrieved_bugs.append(
                        RetrievedBug(
                            dataset_source=record.get(
                                "dataset_source",
                                "Historical Dataset",
                            ),
                            bug_type=record.get(
                                "bug_type"
                            ),
                            bug_description=record.get(
                                "bug_description"
                            ),
                            solution=record.get(
                                "solution"
                            ),
                            similarity=round(
                                item.get(
                                    "similarity",
                                    0,
                                )
                                * 100,
                                1,
                            ),
                            historical_file=(
                                record.get("file")
                                or record.get("filename")
                                or record.get("historical_file")
                            ),
                            historical_code=(
                                record.get("code")
                                or record.get("buggy_code")
                                or record.get("historical_code")
                            ),
                            historical_patch=(
                                record.get("patch")
                                or record.get("diff")
                            ),
                            artifact_type=(
                                record.get("artifact_type")
                                or "dataset"
                            ),
                            evidence=record.get("evidence"),
                        )
                    )
                # ============================================
                # GITHUB ISSUE / PR
                # ============================================
                else:
                    artifact_type = record.get(
                        "artifact_type",
                        "GitHub Artifact",
                    )
                    number = record.get(
                        "number"
                    )
                    title = record.get(
                        "title",
                        "",
                    )
                    body = record.get(
                        "body",
                        "",
                    )
                    description_parts = [
                        artifact_type.replace(
                            "_",
                            " "
                        ).title()
                    ]
                    if number is not None:
                        description_parts.append(
                            f"#{number}"
                        )
                    if title:
                        description_parts.append(
                            title
                        )
                    if body:
                        description_parts.append(
                            body
                        )
                    github_description = (
                        " - ".join(
                            description_parts
                        )
                    )
                    patch = record.get(
                        "patch",
                        "",
                    )
                    retrieved_bugs.append(
                        RetrievedBug(
                            dataset_source=(
                                "GitHub "
                                + artifact_type.replace(
                                    "_",
                                    " ",
                                ).title()
                            ),
                            bug_type=artifact_type,
                            bug_description=(
                                github_description
                            ),
                            solution=(
                                patch[:5000]
                                if patch
                                else None
                            ),
                            similarity=round(
                                item.get(
                                    "similarity",
                                    0,
                                )
                                * 100,
                                1,
                            ),
                            historical_file=(
                                record.get("file")
                                or record.get("filename")
                                or record.get("path")
                            ),
                            historical_code=(
                                record.get("source")
                                or record.get("code")
                                or record.get("historical_code")
                            ),
                            historical_patch=(
                                patch[:10000]
                                if patch
                                else None
                            ),
                            artifact_type=artifact_type,
                            artifact_number=number,
                            artifact_title=title or None,
                            artifact_url=(
                                record.get("url")
                                or record.get("html_url")
                            ),
                            artifact_state=record.get("state"),
                            evidence=body or None,
                        )
                    )
            # =================================================
            # FINAL BUG REPORT
            # =================================================
            new_bug = BugReport(
                    id=str(
                        uuid.uuid4()
                    )[:8],
                    number=len(
                        bug_reports
                    ) + 1,
                    error=analysis.get(
                        "error",
                        finding.get(
                            "error",
                            "Possible issue",
                        ),
                    ),
                    bug_type=bug_type,
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
                        else (
                            "Exact line could not "
                            "be determined."
                        )
                    ),
                    cause=analysis.get(
                        "cause", "",
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
                    llm_unavailable=bool(
                        analysis.get("llm_unavailable", False)
                    ),
                    confidence=confidence,
                    confidence_level=(
                        confidence_level
                    ),
                    confidence_status=(
                        confidence_status
                    ),
                    retrieved_bugs=(
                        retrieved_bugs
                    ),
                    insufficient_evidence=bool(
                        analysis.get(
                            "insufficient_evidence",
                            False,
                        )
                    ),
                    detection_source=finding.get(
                        "detection_source"
                    ) or finding.get("rule"),
                    fix_validated=bool(
                        analysis.get(
                            "fix_validated",
                            False,
                        )
                    ),
                    validation_status=analysis.get(
                        "validation_status"
                    ),
                    validation_message=analysis.get(
                        "validation_message"
                    ),
                    validation_method=analysis.get(
                        "validation_method"
                    ),
                    validation_language=analysis.get(
                        "validation_language"
                    ),
                )
            print(f"[rag] BugReport created with {len(new_bug.retrieved_bugs)} retrieved_bugs")
            bug_reports.append(new_bug)
            # Stream this completed bug to the frontend immediately.
            _push_completed_bug(scan_id, _model_to_dict(new_bug))
        # ====================================================
        # 8. OVERALL CONFIDENCE
        # ====================================================
        overall_confidence = (
            _compute_confidence(
                bug_reports
            )
        )
        # ====================================================
        # 9. FINAL RESPONSE
        # ====================================================

        _update_scan_progress(
            scan_id,
            phase="Finalizing scan",
            progress=99,
            files_processed=len(files),
            files_total=len(files),
            current_file="",
            findings_processed=len(pending),
            findings_total=len(pending),
        )

        return ScanResult(
            summary=ScanSummary(
                repo=(
                    f"{status.owner}/"
                    f"{status.name}"
                ),
                files_scanned=len(
                    files
                ),
                bugs_found=len(
                    bug_reports
                ),
                confidence=(
                    overall_confidence
                ),
                confidence_level=(
                    "High Confidence"
                    if overall_confidence >= 70
                    else "Low Confidence"
                ),
                error_level=(
                    _compute_error_level(
                        bug_reports
                    )
                ),
                scan_status="Completed",
                ai_notice=ai_notice,
                total_files=len(files),
                files_completed=len(files),
                files_failed=0,
                progress_percent=100,
                llm_files_analyzed=len(files),
                llm_findings_analyzed=len(pending),
                rag_enabled=req.rag_enabled,
                rag_records_available=(
                    len(historical_artifacts)
                ),
            ),
            bugs=bug_reports,
        )
    finally:
        # ====================================================
        # CLEANUP
        # ====================================================
        if repo_path:
            cleanup(
                repo_path
            )
# ============================================================
# ASYNC SCAN ENDPOINTS
# ============================================================

@app.post("/api/scan")
def start_scan(req: ScanRequest):
    """
    Start a scan in the background and immediately return a scan_id.
    Up to 4 scans can run concurrently; the executor never blocks.
    """
    scan_id = str(uuid.uuid4())

    _new_scan_job(
        scan_id,
        req.repo_url,
    )

    future = SCAN_JOB_EXECUTOR.submit(
        _run_scan_job,
        scan_id,
        req,
    )

    # If submission raised (shouldn't with max_workers=4 but be safe)
    if future is None:
        with SCAN_JOBS_LOCK:
            job = SCAN_JOBS.get(scan_id)
            if job:
                job["status"] = "failed"
                job["error"] = "Executor rejected the job."

    return {
        "scan_id": scan_id,
        "status": "queued",
        "message": "Scan started.",
    }


@app.get("/api/scan/{scan_id}/bugs")
def scan_bugs_stream(scan_id: str, cursor: int = 0):
    """
    Return completed bugs since `cursor`.

    The frontend polls this alongside the progress endpoint.
    `cursor` is the number of bugs already received; the endpoint
    returns only the new ones beyond that index so the frontend
    can append them one at a time as each LLM analysis finishes.
    """
    job = _get_scan_job(scan_id)

    if job is None:
        return {
            "scan_id": scan_id,
            "status": "not_found",
            "bugs": [],
            "total": 0,
        }

    # Take a snapshot of completed bugs under the lock so we
    # don't hand out a partially-written list.
    with SCAN_JOBS_LOCK:
        raw = SCAN_JOBS.get(scan_id)
        all_bugs = list(raw["completed_bugs"]) if raw else []

    new_bugs = all_bugs[cursor:]

    return {
        "scan_id": scan_id,
        "status": job["status"],
        "bugs": new_bugs,
        "total": len(all_bugs),
        "cursor": cursor + len(new_bugs),
    }


@app.get("/api/scan/{scan_id}")
def scan_progress(scan_id: str):
    """Return live progress or the final scan result."""
    job = _get_scan_job(scan_id)

    if job is None:
        return {
            "scan_id": scan_id,
            "status": "not_found",
            "message": "Scan ID not found.",
        }

    response = {
        "scan_id": job["scan_id"],
        "repo_url": job["repo_url"],
        "status": job["status"],
        "phase": job["phase"],
        "progress": job["progress"],
        "files_processed": job["files_processed"],
        "files_total": job["files_total"],
        "current_file": job["current_file"],
        "findings_processed": job["findings_processed"],
        "findings_total": job["findings_total"],
    }

    if job["error"]:
        response["error"] = job["error"]

    if job["result"] is not None:
        response["result"] = job["result"]

    return response


# ============================================================
# SERVE FRONTEND
# ============================================================
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
