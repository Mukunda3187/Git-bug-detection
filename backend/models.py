"""
Shared data models for the whole backend.
Every module imports from here so the JSON shape sent to the frontend
never drifts between files.
"""
from typing import Optional, List
from pydantic import BaseModel


class ScanRequest(BaseModel):
    repo_url: str


class RepoStatus(BaseModel):
    status: str          # "valid" | "invalid" | "private" | "not_found" | "unreachable"
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
    kind: str = "bug"              # "bug" | "unnecessary_code"
    error: str
    bug_type: str
    status_category: str           # Easy / Frequent / Complex / API / Runtime / Logic / Syntax / Type / Dependency / Security / Performance / Other
    severity: str                  # Low / Medium / High / Critical
    confidence: int                # 0-100
    file: str
    function: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    line_note: Optional[str] = None   # "Exact line could not be determined." when line is unknown
    cause: str
    current_code: str
    replacement_code: Optional[str] = None
    explanation: Optional[str] = None
    action: Optional[str] = None       # for unnecessary code: "Remove this code."
    retrieved_bugs: List[RetrievedBug] = []
    insufficient_evidence: bool = False


class ScanSummary(BaseModel):
    repo: str
    files_scanned: int
    bugs_found: int
    high_severity: int
    medium_severity: int
    low_severity: int
    confidence_high: int
    confidence_medium: int
    confidence_low: int
    scan_status: str


class ScanResult(BaseModel):
    summary: ScanSummary
    bugs: List[BugReport]
