"""
Entry point for the backend. Run with:
    uvicorn main:app --reload --port 8000

Flow per request (matches the abstract exactly):
GitHub URL -> validate -> download -> scan files -> detect candidates
-> RAG retrieve similar historical bugs -> LLM analyze -> build report
"""
import os
import uuid

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from models import ScanRequest, ScanResult, ScanSummary, BugReport, RetrievedBug
from github_handler import check_repository, download_repository, cleanup
from file_scanner import find_source_files, read_file_safely
from detectors.python_detector import detect as detect_python
from rag.retriever import retrieve_similar_bugs
from llm_client import analyze_finding

app = FastAPI(title="RAG-Enhanced LLM for GitHub Bug Detection and Recovery")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DETECTORS_BY_EXTENSION = {
    ".py": detect_python,
    # Other extensions (.js, .java, etc.) are scanned for supported-file
    # listing per the abstract, but only Python has a real AST detector
    # in this build. Add more detectors here as detectors/<lang>_detector.py.
}
MAX_FILES_TO_SCAN = 40          # keeps a single scan fast enough to not hit a gateway timeout
MAX_LLM_CALLS_PER_SCAN = 15     # LLM calls are the slow part - cap them per scan

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
        # Return an empty-but-valid ScanResult shape with the status message
        # in scan_status so the frontend can show it without a special error path.
        return ScanResult(
            summary=ScanSummary(
                repo=req.repo_url, files_scanned=0, bugs_found=0,
                high_severity=0, medium_severity=0, low_severity=0,
                confidence_high=0, confidence_medium=0, confidence_low=0,
                scan_status=status.message,
            ),
            bugs=[],
        )

    repo_path = None
    try:
        repo_path = download_repository(status.owner, status.name, status.default_branch)
    except RuntimeError as e:
        return ScanResult(
            summary=ScanSummary(
                repo=req.repo_url, files_scanned=0, bugs_found=0,
                high_severity=0, medium_severity=0, low_severity=0,
                confidence_high=0, confidence_medium=0, confidence_low=0,
                scan_status=f"Unable to access this repository. Please try again later or check the repository link. ({e})",
            ),
            bugs=[],
        )

       try:
        files = find_source_files(repo_path)
        files_to_scan = files[:MAX_FILES_TO_SCAN]
        bug_reports = []
        llm_calls_used = 0

        for full_path in files_to_scan:
            ext = os.path.splitext(full_path)[1]
            detector = DETECTORS_BY_EXTENSION.get(ext)
            if not detector:
                continue

            source = read_file_safely(full_path)
            if not source:
                continue

            relative_path = os.path.relpath(full_path, repo_path)

            try:
                findings = detector(relative_path, source)
            except Exception:
                # A single bad file should never take down the whole scan.
                continue

            for finding in findings:
                if llm_calls_used >= MAX_LLM_CALLS_PER_SCAN:
                    break
                llm_calls_used += 1
                retrieved = retrieve_similar_bugs(
                    query_text=f"{finding.get('error')}\n{finding.get('current_code')}",
                    top_k=3,
                )
                analysis = analyze_finding(finding, retrieved)

                is_unnecessary = finding.get("rule") == "possibly_unused_function"

                bug_reports.append(BugReport(
                    id=str(uuid.uuid4())[:8],
                    kind="unnecessary_code" if is_unnecessary else "bug",
                    error=analysis.get("error", finding.get("error", "Possible issue")),
                    bug_type=analysis.get("bug_type", finding.get("bug_type", "Other")),
                    status_category=analysis.get("status_category", finding.get("bug_type", "Other")),
                    severity=analysis.get("severity", "Medium"),
                    confidence=int(analysis.get("confidence", 50)),
                    file=relative_path,
                    function=finding.get("function"),
                    line_start=finding.get("line_start"),
                    line_end=finding.get("line_end"),
                    line_note=None if finding.get("line_start") else "Exact line could not be determined.",
                    cause=analysis.get("cause", finding.get("cause", "")),
                    current_code=finding.get("current_code", ""),
                    replacement_code=analysis.get("replacement_code"),
                    explanation=analysis.get("explanation"),
                    action=analysis.get("action"),
                    retrieved_bugs=[
                        RetrievedBug(
                            dataset_source=r["record"].get("dataset_source", "Unknown"),
                            bug_type=r["record"].get("bug_type"),
                            bug_description=r["record"].get("bug_description"),
                            solution=r["record"].get("solution"),
                            similarity=round(r["similarity"] * 100, 1),
                        )
                        for r in retrieved
                    ],
                    insufficient_evidence=bool(analysis.get("insufficient_evidence", False)),
                ))

        high = sum(1 for b in bug_reports if b.severity == "High" or b.severity == "Critical")
        medium = sum(1 for b in bug_reports if b.severity == "Medium")
        low = sum(1 for b in bug_reports if b.severity == "Low")
        conf_high = sum(1 for b in bug_reports if b.confidence >= 80)
        conf_medium = sum(1 for b in bug_reports if 50 <= b.confidence < 80)
        conf_low = sum(1 for b in bug_reports if b.confidence < 50)

        return ScanResult(
            summary=ScanSummary(
                repo=f"{status.owner}/{status.name}",
                files_scanned=len(files_to_scan),
                bugs_found=len(bug_reports),
                high_severity=high,
                medium_severity=medium,
                low_severity=low,
                confidence_high=conf_high,
                confidence_medium=conf_medium,
                confidence_low=conf_low,
                scan_status="Completed",
            ),
            bugs=bug_reports,
        )
    finally:
        if repo_path:
            cleanup(repo_path)


# Serve the frontend as static files so the whole app can run from one process.
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
