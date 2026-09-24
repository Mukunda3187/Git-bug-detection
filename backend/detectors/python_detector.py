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


def detect(file_path: str, source: str):
    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError as e:
        lines = source.splitlines()
        error_line = max(0, min((e.lineno or 1) - 1, max(0, len(lines) - 1)))

        # Include the surrounding indented block instead of only the error line.
        # This gives the LLM enough context to repair missing brackets/colons
        # without inventing the rest of the function.
        start = error_line
        while start > 0:
            previous = lines[start - 1]
            current = lines[start] if start < len(lines) else ""
            if previous.strip() and (len(previous) - len(previous.lstrip()) <=
                                     len(current) - len(current.lstrip())):
                break
            start -= 1

        end = error_line
        base_indent = len(lines[error_line]) - len(lines[error_line].lstrip()) if lines else 0
        while end + 1 < len(lines):
            nxt = lines[end + 1]
            if nxt.strip() and len(nxt) - len(nxt.lstrip()) < base_indent:
                break
            end += 1

        context = "\n".join(lines[start:end + 1]).strip()
        return [{
            "file": file_path,
            "function": None,
            "line_start": start + 1,
            "line_end": end + 1,
            "rule": "syntax_error",
            "error": "Syntax Error",
            "bug_type": "Syntax Error",
            "current_code": context or (lines[error_line].strip() if lines else ""),
            "cause": str(e.msg),
        }]

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
                "current_code": _get_source_segment(source_lines, node)[:200],
                "cause": f"'{name}' is not called anywhere else in this file. It may be used from "
                         f"another file (this detector only sees one file at a time), or it may be dead code.",
            })

    return findings
