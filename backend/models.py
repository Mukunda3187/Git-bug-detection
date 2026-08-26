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
    number: int                    # sequential display number: Bug 1, Bug 2, ...
    kind: str = "bug"              # "bug" | "unnecessary_code"
    error: str                     # short title, e.g. "Possibly unused function"
    bug_type: str
    file: str
    function: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    line_note: Optional[str] = None   # "Exact line could not be determined." when line is unknown
    cause: str
    why_occurs: Optional[str] = None
    solution_type: str = "replace"    # "replace" | "add" | "remove" | "create_file"
    solution_intro: str = ""          # required first sentence, e.g. "Replace the given code with the new code shown below."
    current_code: str
    replacement_code: Optional[str] = None
    add_location: Optional[str] = None   # human description of where to add code, for solution_type == "add"
    new_file_path: Optional[str] = None  # for solution_type == "create_file"
    action: str = ""                     # final one-line action sentence
    explanation: Optional[str] = None
    retrieved_bugs: List[RetrievedBug] = []
    insufficient_evidence: bool = False


class ScanSummary(BaseModel):
    repo: str
    files_scanned: int
    bugs_found: int
    unnecessary_code_found: int
    error_level: str           # "Less Errors" | "Medium Errors" | "More Errors"
    scan_status: str
    ai_notice: Optional[str] = None   # shown when the AI hit a usage limit during this scan


class ScanResult(BaseModel):
    summary: ScanSummary
    bugs: List[BugReport]
