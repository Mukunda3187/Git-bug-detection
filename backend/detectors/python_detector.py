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


def _balance_line_delimiters(line):
    """Return a syntax-balanced version of a simple line with mismatched delimiters.

    This is deliberately quote-aware and only repairs bracket structure; it does
    not try to infer application logic. For example, ``print(sum([1, 2)`` becomes
    ``print(sum([1, 2]))`` by inserting the missing ``]`` and final ``)``.
    """
    opening = "([{"
    closing = ")] }".replace(" ", "")
    pair = {"(": ")", "[": "]", "{": "}"}
    reverse = {v: k for k, v in pair.items()}
    stack = []
    output = []
    quote = None
    triple = False
    escaped = False
    i = 0

    while i < len(line):
        ch = line[i]
        if quote:
            output.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif triple and line.startswith(quote * 3, i):
                output.extend(line[i + 1:i + 3])
                i += 2
                quote = None
                triple = False
            elif not triple and ch == quote:
                quote = None
            i += 1
            continue

        if ch in ("'", '"'):
            quote = ch
            triple = line.startswith(ch * 3, i)
            output.append(ch)
            if triple:
                output.extend(line[i + 1:i + 3])
                i += 2
            i += 1
            continue

        if ch in opening:
            stack.append(ch)
            output.append(ch)
        elif ch in closing:
            if stack and stack[-1] == reverse[ch]:
                stack.pop()
                output.append(ch)
            elif reverse[ch] in stack:
                # Close nested unmatched openers before this closer.
                while stack and stack[-1] != reverse[ch]:
                    output.append(pair[stack.pop()])
                if stack:
                    stack.pop()
                    output.append(ch)
            else:
                # A closer with no opener on this line cannot be repaired safely.
                return None
        else:
            output.append(ch)
        i += 1

    if quote:
        return None
    while stack:
        output.append(pair[stack.pop()])
    fixed = "".join(output)
    return fixed if fixed != line else None


def _syntax_repair_candidate(lines, error):
    """Return (line_index, scratch_line, explanation, display_replacement).

    `scratch_line` is used only to let the parser continue and discover later
    errors. `display_replacement` is the suggested edit shown to the user; it
    may be None when the correct behavior cannot be inferred safely.
    """
    message = str(error.msg or "")
    idx = max(0, min((error.lineno or 1) - 1, max(0, len(lines) - 1)))
    current = lines[idx] if lines else ""
    stripped = current.rstrip()

    # A required parameter cannot follow a parameter with a default.
    if "non-default argument follows default argument" in message.lower() or "parameter without a default follows parameter with a default" in message.lower():
        signature = re.match(r"^(\s*(?:async\s+)?def\s+\w+\s*\()([^()\n]*)(\)\s*:\s*)$", current)
        if signature:
            parts = [part.strip() for part in signature.group(2).split(",")]
            if parts and all(part and not any(ch in part for ch in (":", "*", "/")) for part in parts):
                required = [part for part in parts if "=" not in part]
                optional = [part for part in parts if "=" in part]
                if required and optional:
                    replacement = signature.group(1) + ", ".join(required + optional) + signature.group(3)
                    return idx, replacement, "Move required parameters before parameters with default values.", replacement

    # Repair a missing indentation level after a compound header.
    if "expected an indented block" in message.lower():
        previous_indent = 0
        for prior_idx in range(idx - 1, -1, -1):
            prior = lines[prior_idx]
            if prior.strip():
                previous_indent = len(prior) - len(prior.lstrip())
                break
        replacement = " " * (previous_indent + 4) + current.lstrip()
        if replacement != current:
            return idx, replacement, "Indent this statement inside the preceding block.", replacement

    # A single '=' is not allowed as a condition comparison.
    if re.match(r"^\s*(?:if|elif|while)\b", current) and re.search(r"(?<![<>=!])=(?!=)", current):
        replacement = re.sub(r"(?<![<>=!])=(?!=)", "==", current, count=1)
        return idx, replacement, "Use '==' for comparison instead of assignment in the condition.", replacement

    # A missing colon between a quoted dictionary key and quoted value.
    dict_pair = re.search(r"([\"'])([A-Za-z_]\w*)\1\s+([\"'])", current)
    if dict_pair and "{" in current and "}" in current:
        replacement = current[:dict_pair.start()] + f"{dict_pair.group(1)}{dict_pair.group(2)}{dict_pair.group(1)}: " + current[dict_pair.end() - 1:]
        return idx, replacement, "Add the missing colon between the dictionary key and its value.", replacement

    # Remove a dangling 'if' at the end of a list comprehension.
    if re.search(r"\bif\s*([\]])", current):
        replacement = re.sub(r"\s+if\s*([\]])", r"\1", current, count=1)
        return idx, replacement, "Remove the incomplete filter from the list comprehension.", replacement

    # Repair a malformed nested exception label such as ``except:`` followed
    # by an indented ``ZeroDivisionError:``. Keep the existing nested body valid
    # while making the conditional explicit; the exact exception policy should
    # be reviewed by the developer.
    if re.match(r"^\s*ZeroDivisionError\s*:\s*$", current) and idx > 0 and re.match(r"^\s*except\s*:\s*$", lines[idx - 1]):
        indent = re.match(r"^\s*", current).group(0)
        replacement = indent + "if True:"
        return idx, replacement, "The exception type is incorrectly written as a nested block; use a valid nested suite and review the except clause.", replacement

    # Invalid ``return`` inside a lambda: lambdas contain an expression, not a
    # return statement. Removing this keyword preserves the expression itself.
    lambda_return = re.match(r"^(\s*.*?\blambda\s+[^:]+:\s*)return\s+(.+?)\s*$", current)
    if lambda_return:
        replacement = lambda_return.group(1) + lambda_return.group(2)
        return idx, replacement, "A lambda must contain an expression, not a return statement; remove 'return'.", replacement

    # Repair mismatched or missing closing delimiters on a single line.
    if "does not match opening parenthesis" in message.lower() or "was never closed" in message.lower():
        balanced = _balance_line_delimiters(current)
        if balanced:
            return idx, balanced, "Balance the brackets and parentheses on this line.", balanced

    # Common missing comma in a simple function parameter list: def f(a b):
    m = re.match(r"^(\s*(?:async\s+)?def\s+\w+\s*\()([^()]*)(\)\s*:\s*)$", current)
    if m:
        params = m.group(2)
        fixed = re.sub(r"(?<=[A-Za-z0-9_])\s+(?=[A-Za-z_]\w*(?:\s*=|\s*,|\s*$))", ", ", params, count=1)
        if fixed != params:
            replacement = m.group(1) + fixed + m.group(3)
            return idx, replacement, "A comma is missing between function parameters.", replacement

        # Duplicate parameter names are rejected by Python's compiler.
        names = re.findall(r"(?<!\*)\b([A-Za-z_]\w*)\b(?=\s*(?:=|,|$))", params)
        seen = set()
        parameter_parts = params.split(",")
        seen = set()
        for part_index, part in enumerate(parameter_parts):
            name_match = re.search(r"^\s*(?:\*{0,2})\s*([A-Za-z_]\w*)", part)
            if not name_match:
                continue
            name = name_match.group(1)
            if name in seen:
                renamed = name + "_2"
                parameter_parts[part_index] = part[:name_match.start(1)] + renamed + part[name_match.end(1):]
                fixed_params = ",".join(parameter_parts)
                replacement = m.group(1) + fixed_params + m.group(3)
                return idx, replacement, f"The parameter name '{name}' is repeated; rename one parameter.", replacement
            seen.add(name)

    # Unterminated ordinary string literals (e.g. name = "Munna).
    if "unterminated string literal" in message.lower():
        quote = '"' if current.count('"') % 2 else "'"
        replacement = current.rstrip() + quote
        return idx, replacement, f"The string is missing its closing {quote} quote.", replacement

    # Unclosed delimiter. Close it on its opening line so later errors can be found.
    unclosed = re.search(r"'([([{])' was never closed", message)
    if unclosed:
        opener = unclosed.group(1)
        close_for = {"(": ")", "[": "]", "{": "}"}
        replacement = current.rstrip() + close_for[opener]
        return idx, replacement, f"Add the missing '{close_for[opener]}' to close the '{opener}'.", replacement

    # Missing comma between arguments in a simple call such as f(a=1, 2).
    if "positional argument follows keyword argument" in message.lower():
        call = re.match(r"^(\s*[\w.]+\s*\()(.*)(\)\s*)$", current)
        if call:
            args = call.group(2)
            parts = [part.strip() for part in args.split(",")]
            keyword_parts = [part for part in parts if "=" in part]
            positional_parts = [part for part in parts if "=" not in part]
            if keyword_parts and positional_parts:
                replacement = call.group(1) + ", ".join(positional_parts + keyword_parts) + call.group(3)
                return idx, replacement, "A positional argument appears after a keyword argument; move positional arguments first.", replacement

    # Incomplete `from module import` statement. The intended symbol is unknown;
    # use a module import as a syntactically valid suggestion, clearly labeled.
    if re.match(r"^\s*from\s+[\w.]+\s+import\s*$", current):
        module = re.search(r"from\s+([\w.]+)\s+import", current).group(1)
        replacement = re.match(r"^\s*", current).group(0) + f"import {module}"
        return idx, replacement, "The import statement has no imported name. Import the module itself or specify the required name.", replacement

    # Illegal control-flow statements outside their required context. A comment
    # lets analysis continue; no behavior-preserving edit can be inferred.
    lowered = message.lower()
    if "'break' outside loop" in lowered:
        replacement = re.match(r"^\s*", current).group(0) + "# TODO: move 'break' inside the intended loop"
        return idx, replacement, "'break' is outside a loop. The suggested comment disables the invalid statement; move it into the intended loop if that behavior is required.", replacement
    if "'continue' not properly in loop" in lowered or "'continue' not supported" in lowered or "'continue' outside loop" in lowered:
        replacement = re.match(r"^\s*", current).group(0) + "# TODO: move 'continue' inside the intended loop"
        return idx, replacement, "'continue' is outside a loop. The suggested comment disables the invalid statement; move it into the intended loop if that behavior is required.", replacement
    if "'return' outside function" in lowered:
        replacement = re.match(r"^\s*", current).group(0) + "# TODO: move 'return' inside the intended function"
        return idx, replacement, "'return' is outside a function. The suggested comment disables the invalid statement; move it into the intended function if that behavior is required.", replacement

    # A malformed function header such as `def greet(name:` can make the
    # parser point at the following line. Repair the nearest preceding header.
    for prior_idx in range(idx, max(-1, idx - 4), -1):
        prior = lines[prior_idx].rstrip()
        if re.match(r"^\s*(?:async\s+)?def\s+\w+\s*\(.*:$", prior):
            before_colon = prior[:-1]
            if before_colon.count("(") > before_colon.count(")"):
                replacement = before_colon + "):"
                return prior_idx, replacement, "Add the missing ')' before the colon in the function definition.", replacement

    # Missing colon after a compound statement, including an inline suite such
    # as `if True print('Hello')`.
    header = re.match(r"^(\s*(?:if|elif|for|while|def|class|with|except|finally|try|else|async\s+for|async\s+def|async\s+with)\b)(.*)$", current)
    if header and not stripped.endswith(":"):
        tail = header.group(2).strip()
        if header.group(1).lstrip().startswith(("if", "elif", "for", "while")):
            # A simple inline body after a condition: split at the first statement-like token.
            inline = re.match(r"^(.+?)\s+(print\s*\(|return\b|pass\b|raise\b|break\b|continue\b|[A-Za-z_]\w*\s*=)", tail)
            if inline:
                condition, body = inline.group(1).rstrip(), tail[inline.end(1):].strip()
                replacement = header.group(1) + " " + condition + ": " + body
                return idx, replacement, "Add a colon before the inline statement in the compound header.", replacement
        replacement = stripped + ":"
        return idx, replacement, "Add the missing ':' at the end of the compound statement header.", replacement

    # Repair a mismatched closer by closing the older opener at its own line.
    mismatch = re.search(r"closing parenthesis '([)\]}])' does not match opening parenthesis '([([{])' on line (\d+)", message)
    if mismatch:
        opener = mismatch.group(2)
        opener_index = int(mismatch.group(3)) - 1
        if 0 <= opener_index < len(lines):
            close_for = {"(": ")", "[": "]", "{": "}"}
            replacement = lines[opener_index].rstrip() + close_for[opener]
            return opener_index, replacement, f"Close the unclosed '{opener}' on line {opener_index + 1}.", replacement

    # Unmatched extra closing delimiter at the end of a line.
    unmatched = re.search(r"unmatched '([)\]}])'", message)
    if unmatched and stripped.endswith(unmatched.group(1)):
        closer = unmatched.group(1)
        opener = {')': '(', ']': '[', '}': '{'}[closer]
        excess = max(1, stripped.count(closer) - stripped.count(opener))
        replacement = stripped[:-excess] if excess <= len(stripped) else stripped
        return idx, replacement, f"Remove {excess} unmatched extra '{closer}' character(s).", replacement

    return None


def _detect_syntax_errors(file_path: str, source: str):
    """Find multiple syntax/compile errors by repairing a scratch copy iteratively."""
    lines = source.splitlines() or [""]
    findings = []
    seen = set()

    for _ in range(40):
        candidate_source = "\n".join(lines)
        try:
            compile(candidate_source, file_path, "exec")
            return sorted(findings, key=lambda item: (item["line_start"], item["line_end"]))
        except SyntaxError as error:
            repair = _syntax_repair_candidate(lines, error)
            if repair is None:
                idx = max(0, min((error.lineno or 1) - 1, len(lines) - 1))
                original = lines[idx]
                key = (idx, original, str(error.msg))
                if key in seen:
                    return findings
                seen.add(key)
                findings.append({
                    "file": file_path, "function": None,
                    "line_start": idx + 1, "line_end": idx + 1,
                    "rule": "syntax_error", "error": "Syntax Error",
                    "bug_type": "Syntax Error", "current_code": original,
                    "replacement_code": None, "cause": str(error.msg),
                })
                # Neutralize only the scratch copy so compilation can expose the next issue.
                lines[idx] = "# " + original.lstrip()
                continue

            idx, scratch_line, explanation, display_replacement = repair
            if not (0 <= idx < len(lines)) or scratch_line == lines[idx]:
                return findings
            original_line = lines[idx]
            key = (idx, original_line, explanation)
            if key in seen:
                return findings
            seen.add(key)
            findings.append({
                "file": file_path, "function": None,
                "line_start": idx + 1, "line_end": idx + 1,
                "rule": "syntax_error", "error": "Syntax Error",
                "bug_type": "Syntax Error", "current_code": original_line,
                "replacement_code": display_replacement,
                "cause": explanation,
            })
            lines[idx] = scratch_line

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
