"""
The one and only shape every dataset gets converted into before it goes
into the vector database. If a dataset doesn't have a field, leave it as
None / "" - never invent data.
"""
from dataclasses import dataclass, asdict, field
from typing import Optional


@dataclass
class NormalizedBug:
    id: str
    dataset_source: str        # "BugsInPy" | "Bugs2Fix" | "RunBugRun"
    language: str
    buggy_code: str
    fixed_code: str
    error: Optional[str] = None
    bug_type: Optional[str] = None
    bug_description: Optional[str] = None
    solution: Optional[str] = None
    repository: Optional[str] = None
    file: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    def embedding_text(self) -> str:
        """
        The single text string that gets embedded for similarity search.
        Combining description + buggy code gives the retriever the best
        chance of matching a newly detected bug.
        """
        parts = []
        if self.bug_description:
            parts.append(self.bug_description)
        if self.error:
            parts.append(f"Error: {self.error}")
        parts.append(self.buggy_code)
        return "\n".join(parts)
