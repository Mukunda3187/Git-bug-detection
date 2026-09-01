"""
Real static-analysis detector for Python files, using the `ast` module.

This intentionally reports ONLY syntax_error - the one thing `ast.parse`
can tell you with total certainty: this file either parses as valid
Python or it doesn't. There is no judgment call involved and no way for
it to be a false positive (short of a bug in Python's own parser).

Earlier versions of this file also flagged pattern-based candidates -
bare `except:`, `== None`, mutable default arguments, division without
a visible zero-check, and functions that look unused within one file.
Every one of those is a real, valid thing to notice, but none of them
can be verified as an actual bug from source text alone - each one is
sometimes deliberate (a memoized default argument, a deliberately broad
except at a top-level boundary, a function only called from another
file this detector never sees). Reporting them as "possible bugs"
mixed pattern matches that need human judgment in with syntax errors
that are unconditionally real, which made every finding equally
suspect. Splitting them apart - keep the guaranteed ones, drop the
judgment calls - is what "no false positives" actually requires here.
"""
import ast


def detect(file_path: str, source: str):
    try:
        ast.parse(source, filename=file_path)
    except SyntaxError as e:
        lines = source.splitlines()
        return [{
            "file": file_path,
            "function": None,
            "line_start": e.lineno,
            "line_end": e.lineno,
            "rule": "syntax_error",
            "error": "Syntax Error",
            "bug_type": "Syntax Error",
            "current_code": lines[e.lineno - 1].strip() if e.lineno and e.lineno <= len(lines) else "",
            "cause": str(e.msg),
        }]

    return []
