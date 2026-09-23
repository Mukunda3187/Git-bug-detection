"""
Shared data models for the whole backend.
Every module imports from here so the JSON shape sent to the frontend
never drifts between files.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class ScanRequest(BaseModel):
    repo_url: str


class RepoStatus(BaseModel):
    status: str
    message: str
    owner: Optional[str] = None
    name: Optional[str] = None
    default_branch: Optional[str] = None


class RetrievedBug(BaseModel):
    dataset_source: str
    bug_type: Optional[str] = None
    bug_description: Optional[str] = None
    solution: Optional[str] = None
    similarity: float


class BugReport(BaseModel):
    id: str
    number: int

    error: str
    bug_type: str
    file: str

    function: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    line_note: Optional[str] = None

    cause: str
    why_occurs: Optional[str] = None

    solution_type: str = "replace"
    solution: str = ""

    current_code: str
    replacement_code: Optional[str] = None

    add_location: Optional[str] = None
    new_file_path: Optional[str] = None

    explanation: Optional[str] = None

    # ---------------------------------------------------------
    # CONFIDENCE
    # ---------------------------------------------------------
    # Final confidence calculated by the backend for THIS bug.
    # This is not calculated by the frontend.
    confidence: int = 70

    # Explicit confidence classification produced by the backend.
    confidence_level: str = "Low Confidence"

    # Explicit interpretation of the confidence result.
    confidence_status: str = "Uncertain Finding"

    # ---------------------------------------------------------
    # RAG EVIDENCE
    # ---------------------------------------------------------
    retrieved_bugs: List[RetrievedBug] = Field(default_factory=list)

    # True when the available evidence is not sufficient to make
    # a reliable conclusion.
    insufficient_evidence: bool = False


class ScanSummary(BaseModel):
    repo: str
    files_scanned: int
    bugs_found: int

    # Overall confidence of the findings in this scan.
    confidence: int

    # Overall confidence classification.
    confidence_level: str = "Low Confidence"

    # Overall scan error level.
    error_level: str

    scan_status: str

    ai_notice: Optional[str] = None


class ScanResult(BaseModel):
    summary: ScanSummary
    bugs: List[BugReport] = Field(default_factory=list)
