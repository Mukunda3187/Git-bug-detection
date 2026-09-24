"""
Entry point for the backend.

Flow:
GitHub URL -> validate -> download -> scan files -> detect candidates
-> RAG retrieve similar historical bugs -> LLM analyze -> build report
"""

import os
import uuid
import concurrent.futures

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from models import ScanRequest, ScanResult, ScanSummary, BugReport, RetrievedBug
from github_handler import check_repository, download_repository, cleanup
from file_scanner import find_source_files, read_file_safely
from detectors.python_detector import detect as detect_python
from detectors.js_detector import detect as detect_js
from detectors.cfamily_detector import detect as detect_cfamily
from rag.retriever import retrieve_similar_bugs
from llm_client import analyze_finding, get_fallback_report, analyze_file


app = FastAPI(title="RAG-Enhanced LLM for GitHub Bug Detection and Recovery")


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
# Maximum number of Gemini analyses running at the same time.
# There is no fixed limit on the total number of bugs analyzed.
MAX_PARALLEL_WORKERS = 2

# Number of files being scanned simultaneously.
MAX_FILE_SCAN_WORKERS = 4

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


def _compute_error_level(bug_reports):
    issue_count = len(bug_reports)

    if issue_count <= 10:
        return "Less Errors"
    elif issue_count <= 30:
        return "Medium Errors"
    else:
        return "More Errors"


def _clamp_confidence(value) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 70

    return max(0, min(100, n))


def _compute_confidence(bug_reports):
    if not bug_reports:
        return 100

    return round(
        sum(b.confidence for b in bug_reports) / len(bug_reports)
    )


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/validate")
def validate_repo(req: ScanRequest):
    return check_repository(req.repo_url)


@app.post("/api/scan", response_model=ScanResult)
def scan_repo(req: ScanRequest):

    # ---------------------------------------------------------
    # STEP 1: Validate repository
    # ---------------------------------------------------------

    status = check_repository(req.repo_url)

    if status.status != "valid":
        return _empty_summary(
            req.repo_url,
            status.message
        )

    repo_path = None

    # ---------------------------------------------------------
    # STEP 2: Download repository
    # ---------------------------------------------------------

    try:
        repo_path = download_repository(
            status.owner,
            status.name,
            status.default_branch
        )

    except RuntimeError as e:
        return _empty_summary(
            req.repo_url,
            (
                "Unable to access this repository. "
                "Please try again later or check the repository link. "
                f"({e})"
            ),
        )

    try:

        # -----------------------------------------------------
        # STEP 3: Find source files
        # -----------------------------------------------------

        files = find_source_files(repo_path)

        # -----------------------------------------------------
        # STEP 4: Scan files
        # -----------------------------------------------------

        def _scan_one_file(full_path):

            ext = os.path.splitext(full_path)[1].lower()

            source = read_file_safely(full_path)

            if not source:
                return []

            relative_path = os.path.relpath(
                full_path,
                repo_path
            )

            try:

                detector = DETECTORS_BY_EXTENSION.get(ext)

                if detector:
                    findings = detector(
                        relative_path,
                        source
                    )
                else:
                    findings = analyze_file(
                        relative_path,
                        source
                    )

            except Exception as e:

                print(
                    f"[scan] Failed to analyze "
                    f"{relative_path}: {e}"
                )

                return []

            return [
                (finding, relative_path)
                for finding in findings
            ]

        # Keep original file order.
        file_findings = [None] * len(files)

        if files:

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=MAX_FILE_SCAN_WORKERS
            ) as pool:

                future_to_index = {
                    pool.submit(
                        _scan_one_file,
                        path
                    ): idx
                    for idx, path in enumerate(files)
                }

                for future in concurrent.futures.as_completed(
                    future_to_index
                ):

                    idx = future_to_index[future]

                    try:
                        file_findings[idx] = future.result()

                    except Exception as e:

                        print(
                            f"[scan] Worker error: {e}"
                        )

                        file_findings[idx] = []

        pending = [
            item
            for findings in file_findings
            if findings
            for item in findings
        ]

        print(
            f"[scan] Files scanned: {len(files)}"
        )

        print(
            f"[scan] Findings detected: {len(pending)}"
        )

        # -----------------------------------------------------
        # STEP 5: RAG + Gemini analysis
        #
        # IMPORTANT:
        # There is NO 8-finding limit anymore.
        #
        # Every finding is sent through:
        #
        # Finding
        #   ↓
        # RAG retrieval
        #   ↓
        # Gemini
        #   ↓
        # Bug report
        #
        # Only 2 analyses run concurrently.
        # -----------------------------------------------------

        def _analyze_one(finding, relative_path):

            query_text = (
                f"{finding.get('error', '')}\n"
                f"{finding.get('current_code', '')}"
            )

            # Retrieve historical bugs.
            retrieved = retrieve_similar_bugs(
                query_text=query_text,
                top_k=3,
            )

            try:

                # Every finding gets a real LLM analysis.
                analysis = analyze_finding(
                    finding,
                    retrieved
                )

            except Exception as e:

                print(
                    f"[llm] Analysis failed for "
                    f"{relative_path}: {e}"
                )

                # Safe fallback if Gemini completely fails.
                analysis = get_fallback_report(
                    finding
                )

            return (
                finding,
                relative_path,
                retrieved,
                analysis
            )

        # -----------------------------------------------------
        # Process ALL findings.
        # No MAX_LLM_CALLS_PER_SCAN.
        # -----------------------------------------------------

        results = [None] * len(pending)

        if pending:

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=MAX_PARALLEL_WORKERS
            ) as pool:

                future_to_index = {
                    pool.submit(
                        _analyze_one,
                        finding,
                        relative_path
                    ): idx
                    for idx, (finding, relative_path)
                    in enumerate(pending)
                }

                for future in concurrent.futures.as_completed(
                    future_to_index
                ):

                    idx = future_to_index[future]

                    try:

                        results[idx] = future.result()

                    except Exception as e:

                        print(
                            f"[analysis] Worker error: {e}"
                        )

                        finding, relative_path = pending[idx]

                        retrieved = []

                        analysis = get_fallback_report(
                            finding
                        )

                        results[idx] = (
                            finding,
                            relative_path,
                            retrieved,
                            analysis
                        )

        # -----------------------------------------------------
        # STEP 6: Build final bug reports
        # -----------------------------------------------------

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
                analysis
            ) = result

            # Show AI usage message only once.
            if (
                ai_notice is None
                and analysis.get("rate_limited")
            ):

                ai_notice = analysis.get(
                    "rate_limit_message"
                )

            bug_number += 1

            # -------------------------------------------------
            # Confidence Check
            # -------------------------------------------------

            confidence = _clamp_confidence(
                analysis.get("confidence")
            )

            confidence_level = analysis.get(
                "confidence_level"
            )

            if confidence_level == "High Confidence":

                confidence_level = "High Confidence"

                confidence_status = (
                    "Potential Bug"
                )

            else:

                confidence_level = "Low Confidence"

                confidence_status = (
                    "Uncertain Finding"
                )

            analysis["confidence"] = confidence

            analysis["confidence_level"] = (
                confidence_level
            )

            analysis["confidence_status"] = (
                confidence_status
            )

            print(
                f"[confidence] Bug {bug_number}: "
                f"{confidence_level} -> "
                f"{confidence}% -> "
                f"{confidence_status}"
            )

            # -------------------------------------------------
            # Bug type
            # -------------------------------------------------

            reported_bug_type = analysis.get(
                "bug_type",
                finding.get(
                    "bug_type",
                    "Other"
                )
            )

            if finding.get("rule") == "unreachable_code":

                reported_bug_type = (
                    "Unreachable Code"
                )

            # -------------------------------------------------
            # Create BugReport
            # -------------------------------------------------

            bug_reports.append(
                BugReport(

                    id=str(uuid.uuid4())[:8],

                    number=bug_number,

                    error=analysis.get(
                        "error",
                        finding.get(
                            "error",
                            "Possible issue"
                        )
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
                        if finding.get("line_start")
                        else
                        "Exact line could not be determined."
                    ),

                    cause=analysis.get(
                        "cause",
                        finding.get(
                            "cause",
                            ""
                        )
                    ),

                    why_occurs=analysis.get(
                        "why_occurs"
                    ),

                    solution_type=analysis.get(
                        "solution_type",
                        "replace"
                    ),

                    solution=analysis.get(
                        "solution",
                        ""
                    ),

                    current_code=finding.get(
                        "current_code",
                        ""
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

                    confidence_level=(
                        confidence_level
                    ),

                    confidence_status=(
                        confidence_status
                    ),

                    retrieved_bugs=[
                        RetrievedBug(

                            dataset_source=r[
                                "record"
                            ].get(
                                "dataset_source",
                                "Unknown"
                            ),

                            bug_type=r[
                                "record"
                            ].get(
                                "bug_type"
                            ),

                            bug_description=r[
                                "record"
                            ].get(
                                "bug_description"
                            ),

                            solution=r[
                                "record"
                            ].get(
                                "solution"
                            ),

                            similarity=round(
                                r["similarity"] * 100,
                                1
                            ),
                        )

                        for r in retrieved
                    ],

                    insufficient_evidence=bool(
                        analysis.get(
                            "insufficient_evidence",
                            False
                        )
                    ),
                )
            )

        # -----------------------------------------------------
        # STEP 7: Final summary
        # -----------------------------------------------------

        confidence = _compute_confidence(
            bug_reports
        )

        error_level = _compute_error_level(
            bug_reports
        )

        return ScanResult(

            summary=ScanSummary(

                repo=(
                    f"{status.owner}/"
                    f"{status.name}"
                ),

                files_scanned=len(files),

                bugs_found=len(
                    bug_reports
                ),

                confidence=confidence,

                confidence_level=(
                    "High Confidence"
                    if confidence >= 70
                    else "Low Confidence"
                ),

                error_level=error_level,

                scan_status="Completed",

                ai_notice=ai_notice,
            ),

            bugs=bug_reports,
        )

    finally:

        if repo_path:

            cleanup(repo_path)


# ---------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------

frontend_dir = os.path.join(
    os.path.dirname(__file__),
    "..",
    "frontend"
)

if os.path.isdir(frontend_dir):

    app.mount(
        "/",
        StaticFiles(
            directory=frontend_dir,
            html=True
        ),
        name="frontend"
    )
