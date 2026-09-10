"""
Real static-analysis detector for Python files, using the `ast` module.

Reports two things, and only things that can be verified with total
certainty from the source text alone - no pattern-matching judgment calls:

1. syntax_error - the one thing `ast.parse` can tell you with total
   certainty: this file either parses as valid Python or it doesn't.

2. unreachable_code - code that appears after an unconditional
   return/raise/break/continue in the same block (including an if/else
   or try/except where EVERY branch always terminates - not just a bare
   return/raise/break/continue directly). This is provably dead: there
   is no code path that can ever reach it, regardless of what any other
   file does, unlike "this function looks unused" (which depends on
   whether some other file calls it - a judgment call, not a fact).

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


def _always_terminates(stmt):
    """
    True if executing this ONE statement is guaranteed to exit the
    enclosing block, no matter what - either directly (a plain
    return/raise/break/continue), or because every possible path through
    it ends that way (an if/else where BOTH branches always terminate,
    or a try/except where the try body AND every except handler always
    terminate). This is what lets "if x: return a  else: return b" behave
    like a single terminator for whatever code follows it.

    Deliberately conservative: an if with no else is never treated as
    terminating (the no-else path falls through), and a try's finally
    block is ignored for this check - a finally that terminates would
    make everything unreachable regardless of the try/except, which is
    a much rarer pattern not worth the added complexity here.
    """
    if isinstance(stmt, TERMINATING_STATEMENTS):
        return True

    if isinstance(stmt, ast.If):
        if not stmt.orelse:
            return False  # no else branch - execution can fall through
        return (
            bool(stmt.body) and _always_terminates(stmt.body[-1])
            and bool(stmt.orelse) and _always_terminates(stmt.orelse[-1])
        )

    if isinstance(stmt, ast.Try):
        if not stmt.handlers:
            return False
        try_ok = bool(stmt.body) and _always_terminates(stmt.body[-1])
        handlers_ok = all(
            bool(h.body) and _always_terminates(h.body[-1])
            for h in stmt.handlers
        )
        return try_ok and handlers_ok

    return False


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

        if _always_terminates(stmt):
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
