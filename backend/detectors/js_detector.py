"""
Real static-analysis detector for JavaScript / JSX / TypeScript / TSX
files. No Node.js, tsc, or any external binary is installed in this
Python-only backend, so this does NOT try to be a full parser the way
`ast` is for Python.

Reports two things, both verifiable with certainty from source text alone,
using ONE combined pass over the source (syntax_balance.check_bracket_balance):

1. Structural syntax errors: unbalanced/mismatched brackets and
   unterminated strings/comments - the exact class of error that shows
   up as "Unexpected token" in a Vite/webpack/esbuild build log.

2. Unreachable code: statements after an unconditional return/throw in
   the same block. Deliberately conservative - see check_bracket_balance's
   docstring for exactly what it does and doesn't flag, and why
   (switch/case fallthrough and semicolon-free ASI-style code are both
   left unflagged rather than risk a false positive). Only checked when
   the file has no structural errors, and shares the same JSX/regex/
   string/comment-aware tokenizer as the structural check - so text
   inside JSX children, comments, strings, and regex literals is
   invisible to it for exactly the same reason it's invisible to
   bracket matching.

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
finding equally suspect. Keeping the guaranteed ones and dropping the
rest is what "no false positives" actually requires here.
"""
from .syntax_balance import check_bracket_balance, JS_CONFIG


def detect(file_path: str, source: str):
    structural, unreachable = check_bracket_balance(source, JS_CONFIG)

    if structural:
        finding = dict(structural[0])
        finding["file"] = file_path
        finding["function"] = None
        return [finding]

    findings = []
    for f in unreachable:
        f = dict(f)
        f["file"] = file_path
        f["function"] = None
        findings.append(f)
    return findings
