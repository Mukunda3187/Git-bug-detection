
"""
Entry point for the Git Bug Detection backend.

Flow:
GitHub URL -> validate -> download selected source files -> detect bugs
-> RAG retrieval -> bounded Gemini analysis -> build report.
"""

import os
import uuid
import concurrent.futures

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
from github_handler import check_repository, download_repository, cleanup
from file_scanner import find_source_files, read_file_safely
from detectors.python_detector import detect as detect_python
from detectors.js_detector import detect as detect_js
from detectors.cfamily_detector import detect as detect_cfamily
from rag.retriever import retrieve_similar_bugs
from llm_client import analyze_finding, get_fallback_report


app = FastAPI(title="RAG-Enhanced LLM for GitHub Bug Detection and Recovery")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    ".cs": detect_cfamily,
    ".go": detect_cfamily,
    ".php": detect_cfamily,
}

# Resource limits for a single scan.
MAX_FINDINGS_TO_ANALYZE = 15
MAX_PARALLEL_WORKERS = 2


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
        sum(bug.confidence for bug in bug_reports) / len(bug_reports)
    )


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/validate")
def validate_repo(req: ScanRequest):
    return check_repository(req.repo_url)


@app.post("/api/scan", response_model=ScanResult)
def scan_repo(req: ScanRequest):
    status = check_repository(req.repo_url)

    if status.status != "valid":
        return _empty_summary(req.repo_url, status.message)

    repo_path = None

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

    try:
        files = find_source_files(repo_path)
        pending = []

        # Detect bugs locally first. Do not send whole files to Gemini.
        for full_path in files:
            extension = os.path.splitext(full_path)[1].lower()
            detector = DETECTORS_BY_EXTENSION.get(extension)

            # Only run detectors that are actually supported.
            if detector is None:
                continue

            source = read_file_safely(full_path)
            if not source:
                continue

            relative_path = os.path.relpath(full_path, repo_path)

            try:
                findings = detector(relative_path, source)
            except Exception as exc:
                print(f"[scan] Detector failed for {relative_path}: {exc}")
                continue

            for finding in findings or []:
                pending.append((finding, relative_path))

        print(f"[scan] Files discovered: {len(files)}")
        print(f"[scan] Findings detected: {len(pending)}")

        if len(pending) > MAX_FINDINGS_TO_ANALYZE:
            print(
                f"[scan] Limiting AI analysis to "
                f"{MAX_FINDINGS_TO_ANALYZE} findings."
            )

        selected_findings = pending[:MAX_FINDINGS_TO_ANALYZE]

        def _analyze_one(finding, relative_path):
            query_text = (
                f"{finding.get('error', '')}\n"
                f"{finding.get('current_code', '')}"
            )

            try:
                retrieved = retrieve_similar_bugs(
                    query_text=query_text,
                    top_k=3,
                )
            except Exception as exc:
                print(f"[rag] Retrieval failed: {exc}")
                retrieved = []

            try:
                analysis = analyze_finding(finding, retrieved)
            except Exception as exc:
                print(f"[llm] Analysis failed for {relative_path}: {exc}")
                analysis = get_fallback_report(finding)

            return finding, relative_path, retrieved, analysis

        results = [None] * len(selected_findings)

        if selected_findings:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=MAX_PARALLEL_WORKERS
            ) as pool:
                future_map = {
                    pool.submit(_analyze_one, finding, relative_path): index
                    for index, (finding, relative_path)
                    in enumerate(selected_findings)
                }

                for future in concurrent.futures.as_completed(future_map):
                    index = future_map[future]

                    try:
                        results[index] = future.result()
                    except Exception as exc:
                        print(f"[analysis] Worker failed: {exc}")
                        finding, relative_path = selected_findings[index]
                        results[index] = (
                            finding,
                            relative_path,
                            [],
                            get_fallback_report(finding),
                        )

        bug_reports = []
        ai_notice = None

        for result in results:
            if result is None:
                continue

            finding, relative_path, retrieved, analysis = result

            if ai_notice is None and analysis.get("rate_limited"):
                ai_notice = analysis.get("rate_limit_message")

            confidence = _clamp_confidence(analysis.get("confidence"))
            confidence_level = analysis.get("confidence_level")

            if confidence_level == "High Confidence":
                confidence_status = "Potential Bug"
            else:
                confidence_level = "Low Confidence"
                confidence_status = "Uncertain Finding"

            bug_type = analysis.get(
                "bug_type",
                finding.get("bug_type", "Other"),
            )

            if finding.get("rule") == "unreachable_code":
                bug_type = "Unreachable Code"

            retrieved_bugs = [
                RetrievedBug(
                    dataset_source=item["record"].get(
                        "dataset_source", "Unknown"
                    ),
                    bug_type=item["record"].get("bug_type"),
                    bug_description=item["record"].get("bug_description"),
                    solution=item["record"].get("solution"),
                    similarity=round(item["similarity"] * 100, 1),
                )
                for item in retrieved
            ]

            bug_reports.append(
                BugReport(
                    id=str(uuid.uuid4())[:8],
                    number=len(bug_reports) + 1,
                    error=analysis.get(
                        "error",
                        finding.get("error", "Possible issue"),
                    ),
                    bug_type=bug_type,
                    file=relative_path,
                    function=finding.get("function"),
                    line_start=finding.get("line_start"),
                    line_end=finding.get("line_end"),
                    line_note=(
                        None
                        if finding.get("line_start")
                        else "Exact line could not be determined."
                    ),
                    cause=analysis.get(
                        "cause", finding.get("cause", "")
                    ),
                    why_occurs=analysis.get("why_occurs"),
                    solution_type=analysis.get("solution_type", "replace"),
                    solution=analysis.get("solution", ""),
                    current_code=finding.get("current_code", ""),
                    replacement_code=analysis.get("replacement_code"),
                    add_location=analysis.get("add_location"),
                    new_file_path=analysis.get("new_file_path"),
                    explanation=analysis.get("explanation"),
                    confidence=confidence,
                    confidence_level=confidence_level,
                    confidence_status=confidence_status,
                    retrieved_bugs=retrieved_bugs,
                    insufficient_evidence=bool(
                        analysis.get("insufficient_evidence", False)
                    ),
                )
            )

        overall_confidence = _compute_confidence(bug_reports)

        status_message = "Completed"
        if len(pending) > len(selected_findings):
            status_message = (
                f"Completed with a limit of {MAX_FINDINGS_TO_ANALYZE} "
                f"findings analyzed out of {len(pending)} detected."
            )

        return ScanResult(
            summary=ScanSummary(
                repo=f"{status.owner}/{status.name}",
                files_scanned=len(files),
                bugs_found=len(bug_reports),
                confidence=overall_confidence,
                confidence_level=(
                    "High Confidence"
                    if overall_confidence >= 70
                    else "Low Confidence"
                ),
                error_level=_compute_error_level(bug_reports),
                scan_status=status_message,
                ai_notice=ai_notice,
            ),
            bugs=bug_reports,
        )

    finally:
        if repo_path:
            cleanup(repo_path)


# Serve frontend when available.
frontend_dir = os.path.join(
    os.path.dirname(__file__),
    "..",
    "frontend",
)

if os.path.isdir(frontend_dir):
    app.mount(
        "/",
        StaticFiles(directory=frontend_dir, html=True),
        name="frontend",
    )
