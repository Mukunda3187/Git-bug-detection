"""
Real static-analysis detector for Python files, using the `ast` module.

Reports two things, and only things that can be verified with total
certainty from the source text alone - no pattern-matching judgment calls:

1. syntax_error - the one thing `ast.parse` can tell you with total
   certainty: this file either parses as valid Python or it doesn't.

2. unreachable_code - code that appears after an unconditional
   return/raise/break/continue in the same block. This is provably dead:
   there is no code path that can ever reach it, regardless of what any
   other file does, unlike "this function looks unused" (which depends
   on whether some other file calls it - a judgment call, not a fact).

Earlier versions of this file also flagged bare `except:`, `== None`,
mutable default arguments, division without a visible zero-check, and
functions that look unused within one file. Every one of those is a
real, valid thing to notice, but none of them can be verified as an
actual bug from source text alone - each is sometimes deliberate (a
memoized default argument, a deliberately broad except at a top-level
boundary, a function only called from another file this detector never
sees). Reporting them as "possible bugs" mixed pattern matches that need
human judgment in with syntax errors that are unconditionally real,
which made every finding equally suspect. Splitting them apart - keep
the guaranteed ones, drop the judgment calls - is what "no false
positives" actually requires here.
"""
import ast

TERMINATING_STATEMENTS = (ast.Return, ast.Raise, ast.Break, ast.Continue)


def _get_source_segment(source_lines, node):
    start = node.lineno - 1
    end = getattr(node, "end_lineno", node.lineno) - 1
    return "\n".join(source_lines[start:end + 1]).strip()


def _find_unreachable_code(body, file_path, source_lines, findings):
    """
    Walks a list of statements (a function body, an if-branch, a loop
    body, etc.). The moment a return/raise/break/continue is seen, every
    statement after it in THIS SAME block is unreachable - report the
    first one and stop (no point flagging every line after it too).
    Still recurses into nested blocks that come BEFORE the terminator,
    since those can have their own unreachable code independently.
    """
    terminated = False
    for stmt in body:
        if terminated:
            findings.append({
                "file": file_path,
                "function": None,
                "line_start": stmt.lineno,
                "line_end": getattr(stmt, "end_lineno", stmt.lineno),
                "rule": "unreachable_code",
                "error": "Unreachable code",
                "bug_type": "Unnecessary Code",
                "current_code": _get_source_segment(source_lines, stmt),
                "cause": "This code comes right after a return, raise, break, or continue "
                         "in the same block, so it can never actually run.",
            })
            return  # one finding per block is enough - don't flag every subsequent line too

        if isinstance(stmt, TERMINATING_STATEMENTS):
            terminated = True

        # Recurse into nested bodies so unreachable code INSIDE an if/for/while/try
        # (that itself comes before any terminator at this level) still gets caught.
        for field in ("body", "orelse", "finalbody"):
            nested = getattr(stmt, field, None)
            if nested:
                _find_unreachable_code(nested, file_path, source_lines, findings)

        for handler in getattr(stmt, "handlers", []):
            _find_unreachable_code(handler.body, file_path, source_lines, findings)


def detect(file_path: str, source: str):
    try:
        tree = ast.parse(source, filename=file_path)
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

    source_lines = source.splitlines()
    findings = []

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            _find_unreachable_code(node.body, file_path, source_lines, findings)
            self.generic_visit(node)

        visit_AsyncFunctionDef = visit_FunctionDef

    Visitor().visit(tree)
    return findings
