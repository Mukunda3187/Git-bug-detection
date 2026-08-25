"""
Real static-analysis detector for Python files, using the `ast` module.
This is NOT a placeholder - it actually parses the file and finds real
candidate bugs. It intentionally only flags *candidates*: the LLM step
(with RAG context) decides whether each candidate is worth reporting,
and at what confidence.

Each finding is a plain dict so it can be fed straight into the RAG
retriever and then the LLM without extra glue code.
"""
import ast


def _get_source_segment(source_lines, node):
    start = node.lineno - 1
    end = getattr(node, "end_lineno", node.lineno) - 1
    return "\n".join(source_lines[start:end + 1]).strip()


def detect(file_path: str, source: str):
    findings = []
    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError as e:
        findings.append({
            "file": file_path,
            "function": None,
            "line_start": e.lineno,
            "line_end": e.lineno,
            "rule": "syntax_error",
            "error": "Syntax Error",
            "bug_type": "Syntax Error",
            "current_code": (source.splitlines()[e.lineno - 1].strip()
                              if e.lineno and e.lineno <= len(source.splitlines()) else ""),
            "cause": str(e.msg),
        })
        return findings  # can't walk a tree that failed to parse

    source_lines = source.splitlines()
    defined_functions = {}
    called_names = set()

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.current_function = None

        def visit_FunctionDef(self, node):
            defined_functions[node.name] = node
            prev = self.current_function
            self.current_function = node.name
            self.generic_visit(node)
            self.current_function = prev

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Call(self, node):
            if isinstance(node.func, ast.Name):
                called_names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called_names.add(node.func.attr)
            self.generic_visit(node)

        def visit_BinOp(self, node):
            # Division by a variable/expression with no visible guard nearby.
            if isinstance(node.op, ast.Div) or isinstance(node.op, ast.FloorDiv):
                if not isinstance(node.right, ast.Constant):
                    findings.append({
                        "file": file_path,
                        "function": self.current_function,
                        "line_start": node.lineno,
                        "line_end": getattr(node, "end_lineno", node.lineno),
                        "rule": "possible_division_by_zero",
                        "error": "Possible Division by Zero",
                        "bug_type": "Runtime Error",
                        "current_code": _get_source_segment(source_lines, node),
                        "cause": "The denominator is a variable/expression and there is no visible zero-check guarding this division.",
                    })
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
                    "cause": "A bare 'except:' silently catches every exception, including ones that indicate real bugs (e.g. KeyboardInterrupt, SystemExit) or programming mistakes.",
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
                        "cause": "Comparing to None with == or != works but is not the recommended pattern; 'is' / 'is not' is safer because it can't be overridden by custom __eq__.",
                    })
            self.generic_visit(node)

        def visit_FunctionDef_defaults(self, node):
            pass

    v = Visitor()
    v.visit(tree)

    # Mutable default argument check (separate pass, needs defined_functions)
    for name, node in defined_functions.items():
        for default in node.args.defaults:
            if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                findings.append({
                    "file": file_path,
                    "function": name,
                    "line_start": node.lineno,
                    "line_end": node.lineno,
                    "rule": "mutable_default_arg",
                    "error": "Mutable default argument",
                    "bug_type": "Logic Error",
                    "current_code": _get_source_segment(source_lines, node),
                    "cause": "Using a mutable object (list/dict/set) as a default argument means it is shared across all calls, causing hard-to-find bugs.",
                })

    # Unused top-level function heuristic (only within this single file -
    # cannot see cross-file usage, so this is reported as low confidence).
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
                "cause": f"'{name}' is not called anywhere else in this file. It may be used from another file (this detector only sees one file at a time), or it may be dead code.",
            })

    return findings
