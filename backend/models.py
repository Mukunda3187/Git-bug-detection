"""
Shared Pydantic models for the backend.

All API responses use these models so the frontend receives
a consistent JSON structure.
"""

from typing import Optional, List, Dict, Any

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
# RAG RESULT
# ============================================================

class RetrievedBug(BaseModel):
    dataset_source: str = "Unknown"

    bug_type: Optional[str] = None

    bug_description: Optional[str] = None

    solution: Optional[str] = None

    similarity: float = 0.0

    # Extra historical information can be stored without
    # breaking the frontend contract.
    historical_file: Optional[str] = None

    historical_code: Optional[str] = None


# ============================================================
# BUG REPORT
# ============================================================

class BugReport(BaseModel):

    id: str

    number: int

    # --------------------------------------------------------
    # LOCATION
    # --------------------------------------------------------

    file: str

    function: Optional[str] = None

    line_start: Optional[int] = None

    line_end: Optional[int] = None

    line_note: Optional[str] = None

    # --------------------------------------------------------
    # BUG INFORMATION
    # --------------------------------------------------------

    error: str

    bug_type: str = "Other"

    cause: str = ""

    why_occurs: Optional[str] = None

    # --------------------------------------------------------
    # SOLUTION
    # --------------------------------------------------------

    solution_type: str = "replace"

    solution: str = ""

    current_code: str = ""

    replacement_code: Optional[str] = None

    add_location: Optional[str] = None

    new_file_path: Optional[str] = None

    explanation: Optional[str] = None

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    confidence: int = 70

    confidence_level: str = "Low Confidence"

    confidence_status: str = "Uncertain Finding"

    insufficient_evidence: bool = False

    # --------------------------------------------------------
    # RAG EVIDENCE
    # --------------------------------------------------------

    retrieved_bugs: List[
        RetrievedBug
    ] = Field(
        default_factory=list
    )

    # --------------------------------------------------------
    # SOURCE
    # --------------------------------------------------------

    detection_source: Optional[str] = None

    # Examples:
    #   local_detector
    #   llm_file_analysis
    #   hybrid

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    fix_validated: bool = False

    validation_message: Optional[str] = None


# ============================================================
# SCAN SUMMARY
# ============================================================

class ScanSummary(BaseModel):

    repo: str

    files_scanned: int = 0

    bugs_found: int = 0

    unnecessary_code_found: int = 0

    # Overall confidence of the scan.
    confidence: int = 100

    confidence_level: str = "High Confidence"

    error_level: str = "Less Errors"

    scan_status: str = "Completed"

    ai_notice: Optional[str] = None

    # --------------------------------------------------------
    # SCAN PROGRESS
    # --------------------------------------------------------

    total_files: int = 0

    files_completed: int = 0

    files_failed: int = 0

    progress_percent: int = 0

    # --------------------------------------------------------
    # LLM INFORMATION
    # --------------------------------------------------------

    llm_files_analyzed: int = 0

    llm_findings_analyzed: int = 0

    # --------------------------------------------------------
    # RAG INFORMATION
    # --------------------------------------------------------

    rag_enabled: bool = True

    rag_records_available: int = 0


# ============================================================
# COMPLETE SCAN RESULT
# ============================================================

class ScanResult(BaseModel):

    summary: ScanSummary

    bugs: List[
        BugReport
    ] = Field(
        default_factory=list
    )


# ============================================================
# BACKGROUND SCAN JOB
# ============================================================

class ScanJobResponse(BaseModel):

    scan_id: str

    status: str

    message: str

    progress: int = 0

    result: Optional[
        ScanResult
    ] = None

    error: Optional[str] = None
