"""
Real static-analysis detector for JavaScript / JSX / TypeScript / TSX
files. No Node.js, tsc, or any external binary is installed in this
Python-only backend, so this does NOT try to be a full parser the way
`ast` is for Python - instead it does two things well:

1. Structural syntax check (syntax_balance.check_bracket_balance): finds
   unbalanced/mismatched brackets and unterminated strings - the exact
   class of error that shows up as "Unexpected token" in a Vite/webpack/
   esbuild build log. This is Step 1 from the plan: catch real build-
   breaking errors first, before anything fancier.

2. A small set of precise, low-false-positive heuristic checks that only
   run when step 1 found nothing (no point flagging style issues in a
   file that won't even build): == / != instead of === / !==, var instead
   of let/const, empty catch blocks, and leftover console.log/debugger
   statements.

Deliberately NOT included yet (would need a real parser to do safely,
e.g. via Node/Babel/TypeScript-compiler subprocess, which is a separate
follow-up step): missing React "key" props in .map(), useEffect
dependency-array issues, unused imports/variables, JSX tag balance.
A regex guess at any of these has a real chance of being wrong, and a
wrong "bug" is worse than no detector at all (see llm_client.py's
insufficient_evidence handling) - so they're left for a proper parser-
based pass rather than shipped as unreliable heuristics.
"""
import re

from .syntax_balance import (
    check_bracket_balance, mask_non_code, JS_CONFIG,
)

MAX_HEURISTIC_FINDINGS_PER_FILE = 5  # keep one noisy file from eating the whole LLM-call budget

LOOSE_EQUALITY_RE = re.compile(r"(?<![=!<>])(==|!=)(?!=)")
VAR_DECLARATION_RE = re.compile(r"\bvar\s+[A-Za-z_$]")
EMPTY_CATCH_RE = re.compile(r"catch\s*(\([^)]*\))?\s*\{\s*\}")
CONSOLE_DEBUG_RE = re.compile(r"\bconsole\.(log|debug)\s*\(")
DEBUGGER_RE = re.compile(r"\bdebugger\b\s*;?")


def _line_number_at(text: str, char_index: int) -> int:
    return text.count("\n", 0, char_index) + 1


def _line_text(lines, line_no):
    return lines[line_no - 1].strip() if 0 < line_no <= len(lines) else ""


def _heuristic_findings(file_path: str, source: str):
    findings = []
    masked = mask_non_code(source, JS_CONFIG)  # comments/strings blanked out, line numbers preserved
    lines = source.splitlines()

    def add(match, rule, error, bug_type, cause):
        if len(findings) >= MAX_HEURISTIC_FINDINGS_PER_FILE:
            return
        line_no = _line_number_at(source, match.start())
        findings.append({
            "file": file_path,
            "function": None,
            "line_start": line_no,
            "line_end": line_no,
            "rule": rule,
            "error": error,
            "bug_type": bug_type,
            "current_code": _line_text(lines, line_no),
            "cause": cause,
        })

    for m in LOOSE_EQUALITY_RE.finditer(masked):
        op = m.group(1)
        add(
            m, "loose_equality",
            f"Loose equality ('{op}') used instead of strict equality",
            "Logic Error",
            f"'{op}' compares values after type coercion (e.g. 0 == \"0\" is true), which is a common "
            f"source of subtle bugs. '{'===' if op == '==' else '!=='}' compares type and value together "
            f"and is the recommended default in modern JS/TS.",
        )

    for m in VAR_DECLARATION_RE.finditer(masked):
        add(
            m, "var_declaration",
            "'var' used instead of 'let' / 'const'",
            "Logic Error",
            "'var' is function-scoped (not block-scoped) and is hoisted, which can leak a variable out "
            "of the block it looks like it belongs to. 'let' (or 'const' if it's never reassigned) avoids "
            "that class of bug.",
        )

    for m in EMPTY_CATCH_RE.finditer(masked):
        add(
            m, "empty_catch_block",
            "Empty catch block",
            "Logic Error",
            "An error is caught here and silently discarded - if this code ever throws, the failure "
            "disappears with no log, no fallback, and no way to know it happened.",
        )

    for m in CONSOLE_DEBUG_RE.finditer(masked):
        add(
            m, "leftover_console_statement",
            "Leftover console.log/debug statement",
            "Unnecessary Code",
            "This looks like a debugging statement left in the code. It's harmless in production but "
            "usually isn't meant to ship, and can leak internal data into the browser console.",
        )

    for m in DEBUGGER_RE.finditer(masked):
        add(
            m, "leftover_debugger_statement",
            "Leftover 'debugger' statement",
            "Unnecessary Code",
            "A 'debugger' statement pauses execution in any browser with devtools open. This is almost "
            "always leftover from debugging and shouldn't ship.",
        )

    return findings


def detect(file_path: str, source: str):
    structural = check_bracket_balance(source, JS_CONFIG)
    if structural:
        # Same reasoning as python_detector.py returning right after a SyntaxError:
        # once the file's structure is broken, further checks on it are noise.
        finding = dict(structural[0])
        finding["file"] = file_path
        finding["function"] = None
        return [finding]

    return _heuristic_findings(file_path, source)
