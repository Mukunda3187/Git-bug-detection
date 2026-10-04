"""
Entry point for the Git Bug Detection backend.

CURRENT ARCHITECTURE
--------------------
GitHub URL
    -> validate repository
    -> create background scan job
    -> return scan_id immediately
    -> background worker:
         -> discover source files
         -> run local detectors
         -> RAG retrieval
         -> Gemini analysis
         -> build final report
    -> frontend polls /api/scan/{scan_id}

IMPORTANT:
There is intentionally NO fixed file-count limit in this file.
The scanner decides which source files belong to the repository.

This version moves the expensive repository scan out of the original
HTTP request so large repositories do not keep the browser request open
until the entire scan finishes.
"""

import os
import uuid
import concurrent.futures
import threading
import traceback

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, BackgroundTasks
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
)

from file_scanner import (
    find_source_files,
    read_file_safely,
)

from detectors.python_detector import detect as detect_python
from detectors.js_detector import detect as detect_js
from detectors.cfamily_detector import detect as detect_cfamily

from rag.retriever import retrieve_similar_bugs

from llm_client import (
    analyze_finding,
    get_fallback_report,
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


# ============================================================
# DETECTORS
# ============================================================

DETECTORS_BY_EXTENSION = {
    ".py": detect_python,
    ".pyi": detect_python,

    ".js": detect_js,
    ".jsx": detect_js,
    ".ts": detect_js,
    ".tsx": detect_js,

    ".java": detect_cfamily,

    ".c": detect_cfamily,
    ".h": detect_cfamily,
    ".cpp": detect_cfamily,
    ".hpp": detect_cfamily,
    ".cc": detect_cfamily,
    ".hh": detect_cfamily,

    ".cs": detect_cfamily,
    ".go": detect_cfamily,
    ".php": detect_cfamily,
}


# ============================================================
# CONFIGURATION
# ============================================================

# Number of Gemini/RAG analysis workers running simultaneously.
#
# Keep this controlled for now.
# Later, when we implement the 5-key worker architecture,
# this can be expanded properly.
MAX_PARALLEL_WORKERS = 2


# ============================================================
# IN-MEMORY SCAN JOB STORE
# ============================================================
#
# Example:
#
# SCAN_JOBS[scan_id] = {
#     "status": "queued",
#     "progress": 0,
#     "message": "Waiting to start...",
#     "repo": "...",
#     "files_total": 0,
#     "files_processed": 0,
#     "findings_detected": 0,
#     "result": None,
#     "error": None,
# }
#
# This is intentionally simple for the first stage.
# A persistent database can be added later.

SCAN_JOBS = {}

SCAN_JOBS_LOCK = threading.Lock()


# ============================================================
# JOB HELPERS
# ============================================================

def _create_job(repo_url: str):
    scan_id = str(uuid.uuid4())

    job = {
        "scan_id": scan_id,
        "repo_url": repo_url,
        "status": "queued",
        "progress": 0,
        "message": "Scan queued.",
        "files_total": 0,
        "files_processed": 0,
        "findings_detected": 0,
        "result": None,
        "error": None,
    }

    with SCAN_JOBS_LOCK:
        SCAN_JOBS[scan_id] = job

    return scan_id


def _update_job(scan_id: str, **updates):
    with SCAN_JOBS_LOCK:
        job = SCAN_JOBS.get(scan_id)

        if job is None:
            return

        job.update(updates)


def _get_job(scan_id: str):
    with SCAN_JOBS_LOCK:
        job = SCAN_JOBS.get(scan_id)

        if job is None:
            return None

        # Return a copy so callers do not modify shared state.
        return dict(job)


# ============================================================
# RESULT HELPERS
# ============================================================

def _empty_summary(repo: str, message: str) -> ScanResult:
    return ScanResult(
        summary=ScanSummary(
            repo=repo,
            files_scanned=0,
            bugs_found=0,
            confidence=100,
            confidence_level="High Confidence",
            error_level="Less Errors",
            scan_status=message,
        ),
        bugs=[],
    )


def _compute_error_level(bug_reports):
    count = len(bug_reports)

    if count <= 10:
        return "Less Errors"

    if count <= 30:
        return "Medium Errors"

    return "More Errors"


def _clamp_confidence(value) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 70

    return max(0, min(100, number))


def _compute_confidence(bug_reports):
    if not bug_reports:
        return 100

    return round(
        sum(
            bug.confidence
            for bug in bug_reports
        ) / len(bug_reports)
    )


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
    return check_repository(req.repo_url)


# ============================================================
# START SCAN
# ============================================================
#
# IMPORTANT:
# This endpoint DOES NOT perform the complete scan.
#
# It creates a background job and immediately returns a scan_id.
#
# This prevents:
#
# Browser
#    -> waits for huge scan
#    -> Render timeout
#    -> 502
#
# Instead:
#
# Browser
#    -> create job
#    -> receive scan_id immediately
#
# Then frontend can poll:
#
# GET /api/scan/{scan_id}
#

@app.post("/api/scan")
def start_scan(
    req: ScanRequest,
    background_tasks: BackgroundTasks,
):
    status = check_repository(req.repo_url)

    if status.status != "valid":
        return {
            "status": "failed",
            "message": status.message,
            "scan_id": None,
        }

    scan_id = _create_job(req.repo_url)

    _update_job(
        scan_id,
        status="queued",
        progress=0,
        message="Scan queued. Repository analysis will start shortly.",
    )

    background_tasks.add_task(
        _run_scan_job,
        scan_id,
        req.repo_url,
    )

    return {
        "scan_id": scan_id,
        "status": "queued",
        "message": "Scan started. Use the scan status endpoint to follow progress.",
    }


# ============================================================
# GET SCAN STATUS
# ============================================================

@app.get("/api/scan/{scan_id}")
def get_scan_status(scan_id: str):
    job = _get_job(scan_id)

    if job is None:
        return {
            "status": "not_found",
            "message": "Scan job was not found.",
            "scan_id": scan_id,
        }

    response = {
        "scan_id": scan_id,
        "status": job["status"],
        "progress": job["progress"],
        "message": job["message"],
        "files_total": job["files_total"],
        "files_processed": job["files_processed"],
        "findings_detected": job["findings_detected"],
    }

    if job["status"] == "completed":
        response["result"] = job["result"]

    if job["status"] == "failed":
        response["error"] = job["error"]

    return response


# ============================================================
# BACKGROUND SCAN
# ============================================================

def _run_scan_job(scan_id: str, repo_url: str):

    repo_path = None

    try:

        # ----------------------------------------------------
        # STARTING
        # ----------------------------------------------------

        _update_job(
            scan_id,
            status="starting",
            progress=2,
            message="Validating repository...",
        )

        status = check_repository(repo_url)

        if status.status != "valid":

            result = _empty_summary(
                repo_url,
                status.message,
            )

            _update_job(
                scan_id,
                status="failed",
                progress=100,
                message=status.message,
                result=result.model_dump(),
                error=status.message,
            )

            return

        # ----------------------------------------------------
        # DOWNLOAD / INDEX REPOSITORY
        # ----------------------------------------------------

        _update_job(
            scan_id,
            status="downloading",
            progress=5,
            message="Accessing repository source files...",
        )

        try:

            repo_path = download_repository(
                status.owner,
                status.name,
                status.default_branch,
            )

        except Exception as exc:

            message = (
                f"Unable to download repository: {exc}"
            )

            _update_job(
                scan_id,
                status="failed",
                progress=100,
                message=message,
                error=message,
            )

            return

        # ----------------------------------------------------
        # FIND ALL SOURCE FILES
        # ----------------------------------------------------

        _update_job(
            scan_id,
            status="scanning_files",
            progress=10,
            message="Discovering source-code files...",
        )

        files = find_source_files(repo_path)

        total_files = len(files)

        _update_job(
            scan_id,
            files_total=total_files,
            files_processed=0,
            progress=10,
            message=f"Found {total_files} source files.",
        )

        print(
            f"[scan:{scan_id}] "
            f"Files discovered: {total_files}"
        )

        # ----------------------------------------------------
        # LOCAL DETECTION
        # ----------------------------------------------------

        pending = []

        for file_index, full_path in enumerate(files, start=1):

            extension = (
                os.path.splitext(full_path)[1]
                .lower()
            )

            detector = DETECTORS_BY_EXTENSION.get(
                extension
            )

            relative_path = os.path.relpath(
                full_path,
                repo_path,
            )

            source = read_file_safely(full_path)

            if not source:
                _update_job(
                    scan_id,
                    files_processed=file_index,
                )
                continue

            # ------------------------------------------------
            # RUN LOCAL DETECTOR WHEN AVAILABLE
            # ------------------------------------------------

            if detector is not None:

                try:

                    findings = detector(
                        relative_path,
                        source,
                    )

                except Exception as exc:

                    print(
                        f"[scan:{scan_id}] "
                        f"Detector failed for "
                        f"{relative_path}: {exc}"
                    )

                    findings = []

                for finding in findings or []:
                    pending.append(
                        (
                            finding,
                            relative_path,
                        )
                    )

            # ------------------------------------------------
            # UPDATE PROGRESS
            # ------------------------------------------------

            if total_files > 0:

                # Local scanning occupies roughly 10-50%.
                scan_progress = (
                    10
                    + int(
                        (file_index / total_files) * 40
                    )
                )

            else:
                scan_progress = 50

            _update_job(
                scan_id,
                files_processed=file_index,
                findings_detected=len(pending),
                progress=min(scan_progress, 50),
                message=(
                    f"Scanning file "
                    f"{file_index}/{total_files}"
                ),
            )

        print(
            f"[scan:{scan_id}] "
            f"Findings detected: {len(pending)}"
        )

        _update_job(
            scan_id,
            progress=50,
            findings_detected=len(pending),
            message=(
                f"Source scan completed. "
                f"{len(pending)} potential findings detected. "
                f"Starting RAG and LLM analysis..."
            ),
        )

        # ----------------------------------------------------
        # LLM + RAG ANALYSIS
        # ----------------------------------------------------

        def _analyze_one(
            finding,
            relative_path,
        ):

            query_text = (
                f"{finding.get('error', '')}\n"
                f"{finding.get('current_code', '')}"
            )

            # -----------------------------------------------
            # RAG
            # -----------------------------------------------

            try:

                retrieved = retrieve_similar_bugs(
                    query_text=query_text,
                    top_k=3,
                )

            except Exception as exc:

                print(
                    f"[rag:{scan_id}] "
                    f"Retrieval failed: {exc}"
                )

                retrieved = []

            # -----------------------------------------------
            # GEMINI
            # -----------------------------------------------

            try:

                analysis = analyze_finding(
                    finding,
                    retrieved,
                )

            except Exception as exc:

                print(
                    f"[llm:{scan_id}] "
                    f"Analysis failed for "
                    f"{relative_path}: {exc}"
                )

                analysis = get_fallback_report(
                    finding
                )

            return (
                finding,
                relative_path,
                retrieved,
                analysis,
            )

        results = [None] * len(pending)

        completed_analysis = 0
        total_findings = len(pending)

        # ----------------------------------------------------
        # PROCESS FINDINGS
        # ----------------------------------------------------
        #
        # There is intentionally no fixed number-of-findings
        # limit here.
        #
        # Every detected finding is submitted for analysis.
        #
        # The executor controls how many API calls happen
        # simultaneously.
        #

        if pending:

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=MAX_PARALLEL_WORKERS
            ) as pool:

                future_map = {
                    pool.submit(
                        _analyze_one,
                        finding,
                        relative_path,
                    ): index

                    for index, (
                        finding,
                        relative_path,
                    ) in enumerate(pending)
                }

                for future in concurrent.futures.as_completed(
                    future_map
                ):

                    index = future_map[future]

                    try:

                        results[index] = (
                            future.result()
                        )

                    except Exception as exc:

                        print(
                            f"[analysis:{scan_id}] "
                            f"Worker failed: {exc}"
                        )

                        finding, relative_path = (
                            pending[index]
                        )

                        results[index] = (
                            finding,
                            relative_path,
                            [],
                            get_fallback_report(
                                finding
                            ),
                        )

                    completed_analysis += 1

                    if total_findings > 0:

                        analysis_progress = (
                            50
                            + int(
                                (
                                    completed_analysis
                                    / total_findings
                                )
                                * 45
                            )
                        )

                    else:
                        analysis_progress = 95

                    _update_job(
                        scan_id,
                        progress=min(
                            analysis_progress,
                            95,
                        ),
                        message=(
                            "AI analysis: "
                            f"{completed_analysis}/"
                            f"{total_findings}"
                        ),
                    )

        # ----------------------------------------------------
        # BUILD BUG REPORTS
        # ----------------------------------------------------

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

            # -----------------------------------------------
            # AI RATE LIMIT NOTICE
            # -----------------------------------------------

            if (
                ai_notice is None
                and analysis.get("rate_limited")
            ):
                ai_notice = analysis.get(
                    "rate_limit_message"
                )

            # -----------------------------------------------
            # CONFIDENCE
            # -----------------------------------------------

            confidence = _clamp_confidence(
                analysis.get("confidence")
            )

            confidence_level = analysis.get(
                "confidence_level"
            )

            if confidence_level == "High Confidence":

                confidence_status = "Potential Bug"

            else:

                confidence_level = "Low Confidence"
                confidence_status = (
                    "Uncertain Finding"
                )

            # -----------------------------------------------
            # BUG TYPE
            # -----------------------------------------------

            bug_type = analysis.get(
                "bug_type",
                finding.get(
                    "bug_type",
                    "Other",
                ),
            )

            if finding.get("rule") == "unreachable_code":

                bug_type = "Unreachable Code"

            # -----------------------------------------------
            # RAG RESULTS
            # -----------------------------------------------

            retrieved_bugs = []

            for item in retrieved:

                record = item.get(
                    "record",
                    {},
                )

                retrieved_bugs.append(
                    RetrievedBug(
                        dataset_source=record.get(
                            "dataset_source",
                            "Unknown",
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
                            ) * 100,
                            1,
                        ),
                    )
                )

            # -----------------------------------------------
            # BUG REPORT
            # -----------------------------------------------

            bug_reports.append(
                BugReport(
                    id=str(uuid.uuid4())[:8],

                    number=len(bug_reports) + 1,

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

                    confidence=confidence,

                    confidence_level=confidence_level,

                    confidence_status=confidence_status,

                    retrieved_bugs=retrieved_bugs,

                    insufficient_evidence=bool(
                        analysis.get(
                            "insufficient_evidence",
                            False,
                        )
                    ),
                )
            )

        # ----------------------------------------------------
        # FINAL RESULT
        # ----------------------------------------------------

        overall_confidence = _compute_confidence(
            bug_reports
        )

        final_result = ScanResult(
            summary=ScanSummary(
                repo=(
                    f"{status.owner}/"
                    f"{status.name}"
                ),

                files_scanned=total_files,

                bugs_found=len(
                    bug_reports
                ),

                confidence=overall_confidence,

                confidence_level=(
                    "High Confidence"
                    if overall_confidence >= 70
                    else "Low Confidence"
                ),

                error_level=_compute_error_level(
                    bug_reports
                ),

                scan_status="Completed",

                ai_notice=ai_notice,
            ),

            bugs=bug_reports,
        )

        # ----------------------------------------------------
        # SAVE RESULT
        # ----------------------------------------------------

        _update_job(
            scan_id,
            status="completed",
            progress=100,
            message="Repository scan completed.",
            result=final_result.model_dump(),
            files_processed=total_files,
            findings_detected=len(pending),
        )

        print(
            f"[scan:{scan_id}] "
            f"Completed successfully."
        )

    except Exception as exc:

        traceback.print_exc()

        message = (
            f"Scan failed: {exc}"
        )

        _update_job(
            scan_id,
            status="failed",
            progress=100,
            message=message,
            error=message,
        )

    finally:

        # ----------------------------------------------------
        # CLEANUP
        # ----------------------------------------------------

        if repo_path:

            try:
                cleanup(repo_path)

            except Exception as exc:

                print(
                    f"[scan:{scan_id}] "
                    f"Cleanup failed: {exc}"
                )


# ============================================================
# FRONTEND
# ============================================================

frontend_dir = os.path.join(
    os.path.dirname(__file__),
    "..",
    "frontend",
)

if os.path.isdir(frontend_dir):

    app.mount(
        "/",
        StaticFiles(
            directory=frontend_dir,
            html=True,
        ),
        name="frontend",
    )
