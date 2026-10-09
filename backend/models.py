"""
Shared data models for the whole backend.

Every module imports from here so the JSON shape sent to the
frontend stays consistent across the complete scanning pipeline.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


# ============================================================
# SCAN REQUEST
# ============================================================

class ScanRequest(BaseModel):
    repo_url: str


# ============================================================
# REPOSITORY STATUS
# ============================================================

class RepoStatus(BaseModel):
    status: str
    message: str

    owner: Optional[str] = None
    name: Optional[str] = None
    default_branch: Optional[str] = None


# ============================================================
# HISTORICAL / RAG EVIDENCE
# ============================================================

class RetrievedBug(BaseModel):
    """
    One historical record retrieved by the RAG system.

    It may come from:
      - local bug dataset
      - GitHub Issue
      - GitHub Pull Request
      - historical patch/fix
    """

    dataset_source: str = "Unknown"

    bug_type: Optional[str] = None

    bug_description: Optional[str] = None

    solution: Optional[str] = None

    similarity: float = 0.0

    # --------------------------------------------------------
    # Historical source information
    # --------------------------------------------------------

    historical_file: Optional[str] = None

    historical_code: Optional[str] = None

    historical_patch: Optional[str] = None

    artifact_type: Optional[str] = None

    artifact_number: Optional[int] = None

    artifact_title: Optional[str] = None

    artifact_url: Optional[str] = None

    artifact_state: Optional[str] = None

    # Optional evidence text used by the frontend/debugging.
    evidence: Optional[str] = None


# ============================================================
# BUG REPORT
# ============================================================

class BugReport(BaseModel):
    id: str
    number: int

    # --------------------------------------------------------
    # LOCATION
    # --------------------------------------------------------

    error: str
    bug_type: str
    file: str

    function: Optional[str] = None

    line_start: Optional[int] = None
    line_end: Optional[int] = None

    line_note: Optional[str] = None

    # --------------------------------------------------------
    # EXPLANATION
    # --------------------------------------------------------

    cause: str = ""

    why_occurs: Optional[str] = None

    explanation: Optional[str] = None

    # --------------------------------------------------------
    # CORRECTION ACTION
    # --------------------------------------------------------
    #
    # Every finding should explicitly be:
    #
    #   replace
    #   remove
    #   add
    #

    solution_type: str = "replace"

    solution: str = ""

    current_code: str = ""

    replacement_code: Optional[str] = None

    add_location: Optional[str] = None

    new_file_path: Optional[str] = None

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    # Final confidence calculated by backend.
    confidence: int = 70

    # High Confidence / Low Confidence.
    confidence_level: str = "Low Confidence"

    # Human-readable interpretation.
    confidence_status: str = "Uncertain Finding"

    # True when available evidence is insufficient.
    insufficient_evidence: bool = False

    # --------------------------------------------------------
    # RAG / HISTORICAL EVIDENCE
    # --------------------------------------------------------

    retrieved_bugs: List[RetrievedBug] = Field(
        default_factory=list
    )

    # --------------------------------------------------------
    # DETECTION / VALIDATION
    # --------------------------------------------------------

    detection_source: Optional[str] = None

    # True when the proposed correction passed the validator.
    #
    # IMPORTANT:
    # This means the correction passed the available
    # syntax/structural validation. It does not mean that
    # logical correctness has been mathematically proven.
    fix_validated: bool = False

    validation_status: Optional[str] = None

    validation_message: Optional[str] = None

    validation_method: Optional[str] = None

    validation_language: Optional[str] = None


# ============================================================
# SCAN SUMMARY
# ============================================================

class ScanSummary(BaseModel):
    repo: str

    files_scanned: int = 0

    bugs_found: int = 0

    unnecessary_code_found: int = 0

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    confidence: int = 70

    confidence_level: str = "Low Confidence"

    # --------------------------------------------------------
    # SCAN STATUS
    # --------------------------------------------------------

    error_level: str = "Less Errors"

    scan_status: str = "Completed"

    ai_notice: Optional[str] = None

    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    total_files: int = 0

    files_completed: int = 0

    files_failed: int = 0

    progress_percent: int = 0

    # --------------------------------------------------------
    # LLM / RAG INFORMATION
    # --------------------------------------------------------

    llm_files_analyzed: int = 0

    llm_findings_analyzed: int = 0

    rag_enabled: bool = True

    rag_records_available: int = 0


# ============================================================
# FINAL SCAN RESULT
# ============================================================

class ScanResult(BaseModel):
    summary: ScanSummary

    bugs: List[BugReport] = Field(
        default_factory=list
    )


# ============================================================
# ASYNC SCAN JOB RESPONSE
# ============================================================

class ScanJobResponse(BaseModel):
    scan_id: str

    status: str

    message: str

    progress: int = 0

    result: Optional[ScanResult] = None

    error: Optional[str] = None