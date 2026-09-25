"""
Static-analysis detector for Python files, using the `ast` module.

Reports every real, source-verifiable issue this file can find in one
pass over the AST - not just the narrow "100%-certain" subset used
earlier in this project. Every finding still carries an accurate `rule`
name so the rest of the pipeline (llm_client.py's confidence table, the
LLM prompt) can give it an honest confidence score rather than pretend
every finding is equally certain:

- syntax_error              - certain (ast.parse itself fails)
- unreachable_code          - certain (provably dead code)
- bare_except               - very likely a real problem, near-certain
- eq_none                   - very likely a real problem, near-certain
- mutable_default_arg       - very likely a real problem, near-certain
- possible_division_by_zero - a real risk, but sometimes deliberately unguarded
- possibly_unused_function  - a genuine heuristic guess (this file alone
                               can't know if another file imports and
                               calls it) - reported with lower confidence
"""
import ast
import re

TERMINATING_STATEMENTS = (ast.Return, ast.Raise, ast.Break, ast.Continue)


def _always_terminates(stmt):
    """
    True if executing this ONE statement is guaranteed to exit the
    enclosing block - either directly (a plain return/raise/break/
    continue), or because every possible path through it ends that way
    (an if/else where BOTH branches always terminate, or a try/except
    where the try body AND every handler always terminate).
    """
    if isinstance(stmt, TERMINATING_STATEMENTS):
        return True

    if isinstance(stmt, ast.If):
        if not stmt.orelse:
            return False
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
            return

        if _always_terminates(stmt):
            terminated = True

        for field in ("body", "orelse", "finalbody"):
            nested = getattr(stmt, field, None)
            if nested:
                _find_unreachable_code(nested, file_path, source_lines, findings)

        for handler in getattr(stmt, "handlers", []):
            _find_unreachable_code(handler.body, file_path, source_lines, findings)


def _syntax_repair_candidate(lines, error):
    """Return (line_index, updated_line, explanation) for safe common repairs."""
    message = str(error.msg or "")
    error_index = max(0, min((error.lineno or 1) - 1, max(0, len(lines) - 1)))
    current = lines[error_index] if lines else ""

    # Repair malformed function signatures even when CPython points at a later
    # line because the open parenthesis makes the parser treat later lines as
    # part of the signature.
    for idx, candidate_line in enumerate(lines[:error_index + 1]):
        candidate = candidate_line.rstrip()
        if re.match(r"^\s*(?:async\s+)?def\s+\w+\s*\(.*:$", candidate):
            before_colon = candidate[:-1]
            if before_colon.count("(") > before_colon.count(")"):
                return idx, before_colon + "):", "Add the missing ')' before the colon in this function definition."

    # A compound statement header missing its trailing colon. CPython uses
    # both "expected ':'" and the broader "invalid syntax" for these cases.
    header_pattern = r"^\s*(?:async\s+def\s+\w+|def\s+\w+|for\b|async\s+for\b|if\b|elif\b|else\b|while\b|class\s+\w+|except\b|try\b|finally\b|with\b|async\s+with\b).*"
    if current.strip() and re.match(header_pattern, current):
        stripped = current.rstrip()
        # `def greet(name:` has a colon but is missing the closing `)`.
        if re.match(r"^\s*(?:async\s+)?def\s+\w+\s*\(.*:$", stripped):
            before_colon = stripped[:-1]
            if before_colon.count("(") > before_colon.count(")"):
                return error_index, before_colon + "):", "Add the missing ')' before the colon in this function definition."
        if not stripped.endswith(":"):
            return error_index, stripped + ":", "Add the missing ':' at the end of this Python statement header."

    # For an unclosed delimiter, Python points to the opener's line.
    unclosed = re.search(r"'([([{])' was never closed", message)
    if unclosed:
        opener = unclosed.group(1)
        close_for = {"(": ")", "[": "]", "{": "}"}
        opener_index = error_index
        if "line " in message:
            # SyntaxError's line is usually already the opener line.
            opener_index = error_index
        return opener_index, lines[opener_index].rstrip() + close_for[opener], f"Add the missing '{close_for[opener]}' to close the '{opener}' opened on this line."

    # Repair a mismatched closer by closing the older opener at its own line.
    mismatch = re.search(r"closing parenthesis '([)\]}])' does not match opening parenthesis '([([{])' on line (\d+)", message)
    if mismatch:
        opener = mismatch.group(2)
        opener_index = int(mismatch.group(3)) - 1
        if 0 <= opener_index < len(lines):
            close_for = {"(": ")", "[": "]", "{": "}"}
            return opener_index, lines[opener_index].rstrip() + close_for[opener], f"Close the unclosed '{opener}' from line {opener_index + 1} before the later mismatched closer."

    # An unmatched extra closing delimiter can be safely removed when it is trailing.
    unmatched = re.search(r"unmatched '([)\]}])'", message)
    if unmatched and current.rstrip().endswith(unmatched.group(1)):
        closer = unmatched.group(1)
        opener_for = {")": "(", "]": "[", "}": "{"}
        opener = opener_for[closer]
        # Remove the number of trailing closers that cannot be paired with an
        # opener on this line. This treats `return x * y))` as one malformed
        # line and produces the actually valid correction `return x * y`.
        excess = max(1, current.count(closer) - current.count(opener))
        updated = current.rstrip()
        removed = 0
        while removed < excess and updated.endswith(closer):
            updated = updated[:-1]
            removed += 1
        return error_index, updated, f"Remove the {removed} unmatched extra '{closer}' character(s) at the end of this line."

    # Common malformed function header: `def greet(name:` -> `def greet(name):`.
    stripped = current.rstrip()
    if re.match(r"^\s*(async\s+)?def\s+\w+\s*\(.*:$", stripped):
        before_colon = stripped[:-1]
        if before_colon.count("(") > before_colon.count(")"):
            return error_index, before_colon + "):", "Add the missing ')' before the colon in this function definition."

    # Sometimes the parser points at the first statement after the malformed header.
    if "invalid syntax" in message or "expected ':'" in message:
        for idx in range(error_index, max(-1, error_index - 3), -1):
            candidate = lines[idx].rstrip()
            if re.match(r"^\s*(async\s+)?def\s+\w+\s*\(.*:$", candidate):
                before_colon = candidate[:-1]
                if before_colon.count("(") > before_colon.count(")"):
                    return idx, before_colon + "):", "Add the missing ')' before the colon in this function definition."

    return None


def _detect_syntax_errors(file_path: str, source: str):
    """Recover from common independent syntax errors and report each repair."""
    lines = source.splitlines()
    if not lines:
        lines = [""]
    findings = []
    seen = set()

    for _ in range(30):
        candidate_source = "\n".join(lines)
        try:
            ast.parse(candidate_source, filename=file_path)
            return sorted(findings, key=lambda item: (item["line_start"], item["line_end"]))
        except SyntaxError as error:
            repair = _syntax_repair_candidate(lines, error)
            if repair is None:
                # Preserve a useful finding when automatic recovery is uncertain.
                idx = max(0, min((error.lineno or 1) - 1, len(lines) - 1))
                original = lines[idx]
                key = (idx, original, str(error.msg))
                if key not in seen:
                    findings.append({
                        "file": file_path, "function": None,
                        "line_start": idx + 1, "line_end": idx + 1,
                        "rule": "syntax_error", "error": "Syntax Error",
                        "bug_type": "Syntax Error", "current_code": original,
                        "replacement_code": None, "cause": str(error.msg),
                    })
                return findings

            idx, updated_line, explanation = repair
            if not (0 <= idx < len(lines)) or updated_line == lines[idx]:
                return findings
            original_line = lines[idx]
            key = (idx, original_line, updated_line)
            if key not in seen:
                seen.add(key)
                findings.append({
                    "file": file_path, "function": None,
                    "line_start": idx + 1, "line_end": idx + 1,
                    "rule": "syntax_error", "error": "Syntax Error",
                    "bug_type": "Syntax Error", "current_code": original_line,
                    "replacement_code": updated_line, "cause": explanation,
                })
            lines[idx] = updated_line

    return findings


def detect(file_path: str, source: str):
    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError:
        return _detect_syntax_errors(file_path, source)

    source_lines = source.splitlines()
    findings = []
    defined_functions = {}
    called_names = set()

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.current_function = None

        def visit_FunctionDef(self, node):
            defined_functions[node.name] = node
            _find_unreachable_code(node.body, file_path, source_lines, findings)

            prev = self.current_function
            self.current_function = node.name

            for default in node.args.defaults:
                if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                    findings.append({
                        "file": file_path,
                        "function": node.name,
                        "line_start": node.lineno,
                        "line_end": node.lineno,
                        "rule": "mutable_default_arg",
                        "error": "Mutable default argument",
                        "bug_type": "Logic Error",
                        "current_code": _get_source_segment(source_lines, node),
                        "cause": "Using a mutable object (list/dict/set) as a default argument means it "
                                 "is shared across every call, which usually isn't what was intended.",
                    })

            self.generic_visit(node)
            self.current_function = prev

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Call(self, node):
            if isinstance(node.func, ast.Name):
                called_names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called_names.add(node.func.attr)
            self.generic_visit(node)

        def visit_ExceptHandler(self, node):
            if node.type is None:
                findings.append({
                    "file": file_path,
                    "function": self.current_function,
                    "line_start": node.lineno,
                    "line_end": getattr(node, "end_lineno", node.lineno),
                    "rule": "bare_except",
                    "error": "Bare except clause",
                    "bug_type": "Logic Error",
                    "current_code": _get_source_segment(source_lines, node),
                    "cause": "A bare 'except:' silently catches every exception, including ones that "
                             "usually indicate a real bug (like KeyboardInterrupt or SystemExit).",
                })
            self.generic_visit(node)

        def visit_Compare(self, node):
            for op, comparator in zip(node.ops, node.comparators):
                if isinstance(op, (ast.Eq, ast.NotEq)) and isinstance(comparator, ast.Constant) and comparator.value is None:
                    findings.append({
                        "file": file_path,
                        "function": self.current_function,
                        "line_start": node.lineno,
                        "line_end": getattr(node, "end_lineno", node.lineno),
                        "rule": "eq_none",
                        "error": "Comparison to None using == / !=",
                        "bug_type": "Logic Error",
                        "current_code": _get_source_segment(source_lines, node),
                        "cause": "Comparing to None with == or != usually works, but 'is'/'is not' is "
                                 "the correct way since it can't be fooled by a custom __eq__.",
                    })
            self.generic_visit(node)

        def visit_BinOp(self, node):
            if isinstance(node.op, (ast.Div, ast.FloorDiv)) and not isinstance(node.right, ast.Constant):
                findings.append({
                    "file": file_path,
                    "function": self.current_function,
                    "line_start": node.lineno,
                    "line_end": getattr(node, "end_lineno", node.lineno),
                    "rule": "possible_division_by_zero",
                    "error": "Possible Division by Zero",
                    "bug_type": "Runtime Error",
                    "current_code": _get_source_segment(source_lines, node),
                    "cause": "The denominator is a variable or expression with no visible check that "
                             "it isn't zero before this division happens.",
                })
            self.generic_visit(node)

    Visitor().visit(tree)

    for name, node in defined_functions.items():
        if name.startswith("_") or name in ("main", "__init__"):
            continue
        if name not in called_names:
            findings.append({
                "file": file_path,
                "function": name,
                "line_start": node.lineno,
                "line_end": node.lineno,
                "rule": "possibly_unused_function",
                "error": "Possibly unused function",
                "bug_type": "Unnecessary Code",
                "current_code": _get_source_segment(source_lines, node),
                "cause": f"'{name}' is not called anywhere else in this file. It may be used from "
                         f"another file (this detector only sees one file at a time), or it may be dead code.",
            })

    return findings
