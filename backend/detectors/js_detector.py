"""
Detector for JavaScript / JSX / TypeScript / TSX files. No Node.js, tsc,
or any external binary is installed in this Python-only backend, so this
does NOT try to be a full parser the way `ast` is for Python.

Reports every issue this file can find in one pass:

1. Structural syntax errors (syntax_balance.check_bracket_balance) -
   unbalanced/mismatched brackets, unterminated strings/comments.

2. Unreachable code - statements after an unconditional return/throw in
   the same block. See check_bracket_balance's docstring for exactly
   what it does and doesn't flag.

3. Pattern-based heuristic checks, run on a version of the source with
   comment/string/template contents masked out (syntax_balance.
   mask_non_code) so a match inside a string or comment doesn't count:
   - loose_equality: == / != instead of === / !==
   - var_declaration: var instead of let/const
   - empty_catch_block: catch {} with nothing inside
   - leftover_console_statement / leftover_debugger_statement: likely
     debugging leftovers

   These are genuine, real patterns worth flagging, but - unlike the
   structural and unreachable-code checks above - they are judgment
   calls, not certainties: a console.log in a backend entrypoint is
   often deliberate operational logging, not a debugging leftover, and
   a static pattern match can't always tell the difference. Every
   finding still carries an honest confidence score reflecting this
   (see llm_client.FALLBACK_CONFIDENCE_BY_RULE) rather than treating a
   guess the same as a proven syntax error.
"""
import re

from .syntax_balance import check_bracket_balance, mask_non_code, JS_CONFIG

LOOSE_EQ_RE = re.compile(r"(?<![=!])(==|!=)(?!=)")
VAR_RE = re.compile(r"\bvar\s+[a-zA-Z_$][\w$]*")
EMPTY_CATCH_RE = re.compile(r"catch\s*(\([^)]*\))?\s*\{\s*\}")
CONSOLE_RE = re.compile(r"\bconsole\s*\.\s*(log|debug|warn|info)\s*\(")
DEBUGGER_RE = re.compile(r"\bdebugger\b\s*;?")


def _line_of(source: str, index: int) -> int:
    return source.count("\n", 0, index) + 1


def _line_text(lines, ln):
    return lines[ln - 1].strip() if 0 < ln <= len(lines) else ""


def _pattern_findings(source: str, file_path: str):
    masked = mask_non_code(source, JS_CONFIG)
    lines = source.splitlines()
    findings = []

    for m in LOOSE_EQ_RE.finditer(masked):
        ln = _line_of(source, m.start())
        findings.append({
            "file": file_path, "function": None,
            "line_start": ln, "line_end": ln,
            "rule": "loose_equality",
            "error": f"Loose equality ({m.group(1)})",
            "bug_type": "Logic Error",
            "current_code": _line_text(lines, ln),
            "cause": f"'{m.group(1)}' compares values after converting their types, which can give "
                     f"surprising results. '{m.group(1)}=' compares without any type conversion.",
        })

    for m in VAR_RE.finditer(masked):
        ln = _line_of(source, m.start())
        findings.append({
            "file": file_path, "function": None,
            "line_start": ln, "line_end": ln,
            "rule": "var_declaration",
            "error": "Use of 'var'",
            "bug_type": "Logic Error",
            "current_code": _line_text(lines, ln),
            "cause": "'var' is function-scoped and can lead to confusing bugs across blocks. "
                     "'let' or 'const' are block-scoped and generally safer.",
        })

    for m in EMPTY_CATCH_RE.finditer(masked):
        ln = _line_of(source, m.start())
        findings.append({
            "file": file_path, "function": None,
            "line_start": ln, "line_end": ln,
            "rule": "empty_catch_block",
            "error": "Empty catch block",
            "bug_type": "Logic Error",
            "current_code": _line_text(lines, ln),
            "cause": "This catch block does nothing, so if an error happens here it is silently "
                     "swallowed with no record of it anywhere.",
        })

    for m in CONSOLE_RE.finditer(masked):
        ln = _line_of(source, m.start())
        findings.append({
            "file": file_path, "function": None,
            "line_start": ln, "line_end": ln,
            "rule": "leftover_console_statement",
            "error": "console statement",
            "bug_type": "Unnecessary Code",
            "current_code": _line_text(lines, ln),
            "cause": "This looks like it might be a debugging statement left in the code. It could "
                     "also be intentional logging - review before removing.",
        })

    for m in DEBUGGER_RE.finditer(masked):
        ln = _line_of(source, m.start())
        findings.append({
            "file": file_path, "function": None,
            "line_start": ln, "line_end": ln,
            "rule": "leftover_debugger_statement",
            "error": "'debugger' statement",
            "bug_type": "Unnecessary Code",
            "current_code": _line_text(lines, ln),
            "cause": "A 'debugger' statement pauses execution in browser dev tools - almost always "
                     "a debugging leftover that shouldn't ship.",
        })

    return findings


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

    findings.extend(_pattern_findings(source, file_path))
    return findings
