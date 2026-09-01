"""
Real static-analysis detector for JavaScript / JSX / TypeScript / TSX
files. No Node.js, tsc, or any external binary is installed in this
Python-only backend, so this does NOT try to be a full parser the way
`ast` is for Python.

This reports ONLY the structural syntax check (syntax_balance.
check_bracket_balance): unbalanced/mismatched brackets and unterminated
strings/comments - the exact class of error that shows up as
"Unexpected token" in a Vite/webpack/esbuild build log. That checker
now cross-checks its own JSX-aware findings against a JSX-blind raw
bracket count and suppresses anything the two disagree on (see
syntax_balance.py) - the closest this can get to a guarantee without
an actual parser.

Earlier versions of this file also flagged pattern-based candidates:
loose equality (==), var instead of let/const, empty catch blocks, and
leftover console.log/debugger statements. Each of those is a real,
valid thing to notice, but none can be verified as an actual bug from
source text alone - a console.log in a backend entrypoint is routinely
deliberate operational logging, not a debugging leftover, and a static
pattern match has no way to tell the difference (confirmed directly:
5 out of 5 console.log flags in one real server.ts turned out to be
intentional). Reporting them as "possible bugs" mixed judgment calls
in with syntax errors that are unconditionally real, which made every
finding equally suspect. Keeping the guaranteed one and dropping the
rest is what "no false positives" actually requires here.
"""
from .syntax_balance import check_bracket_balance, JS_CONFIG


def detect(file_path: str, source: str):
    structural = check_bracket_balance(source, JS_CONFIG)
    if not structural:
        return []

    finding = dict(structural[0])
    finding["file"] = file_path
    finding["function"] = None
    return [finding]
