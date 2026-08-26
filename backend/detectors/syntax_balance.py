"""
Language-agnostic structural syntax checker shared by every C-family-style
detector (JavaScript/JSX, TypeScript/TSX, Java, C, C++, C#, Go, PHP —
basically anything that uses (), [], {} for structure, "..."/'...' for
strings, and // + /* */ for comments).

This is NOT a compiler and does not understand grammar. It does exactly
one job, well: walk the source character-by-character, correctly skip
over comments and string/template literals, and confirm every bracket
that opens is closed, in the right order, before the file ends.

That one job is exactly what caught (or rather, what the project's old
Python-only detector *didn't* catch) this real failure:

    [builtin:vite-transform] Unexpected token
        [ src/App.jsx:36:1 ]
        36 | }

Real parsers (Vite/esbuild, tsc, javac, gcc, go build, php -l) all catch
this too, but none of them are installed in this Python-only backend —
this reproduces the same signal without needing any of them.

Design note: like python_detector.py's `ast.parse`, this returns on the
FIRST hard structural error it finds instead of continuing. Once one
bracket is out of sync, every closing bracket after it looks "wrong" too
(hundreds of false positives from one root cause) — so, same as a real
parser bailing out at the first syntax error in a file, we report just
the one true root cause.
"""
from dataclasses import dataclass


@dataclass
class LangConfig:
    string_quotes: str = "'\""          # characters that start/end a normal, single-line string
    template_quote: str = None          # e.g. "`" for JS/TS/Go; None if the language has none
    template_has_interpolation: bool = True   # JS `${expr}` — Go raw strings do NOT have this
    template_escapes: bool = True             # JS backslash-escapes inside strings — Go raw strings do NOT
    multiline_strings_allowed: bool = False   # True only for languages whose *normal* quotes can span lines


JS_CONFIG = LangConfig(template_quote="`", template_has_interpolation=True, template_escapes=True)
GO_CONFIG = LangConfig(template_quote="`", template_has_interpolation=False, template_escapes=False)
C_FAMILY_CONFIG = LangConfig(template_quote=None)  # Java, C, C++, C#, PHP

BRACKET_OPENERS = "([{"
BRACKET_CLOSERS = ")]}"
MATCH = {"(": ")", "[": "]", "{": "}"}


def _line_text(lines, ln):
    return lines[ln - 1].strip() if 0 < ln <= len(lines) else ""


def check_bracket_balance(source: str, config: LangConfig = JS_CONFIG):
    """
    Returns a list with 0 or 1 finding dicts. Empty list = brackets,
    strings, and comments all balance correctly (this file is not the
    cause of a bundler/compiler "unexpected token"-style failure).
    """
    lines = source.splitlines()
    stack = []              # dicts: char, line, col, is_template_expr
    mode = "code"           # code | line_comment | block_comment | string | template
    string_quote_char = None
    line, col = 1, 1
    i, n = 0, len(source)

    while i < n:
        c = source[i]
        nxt = source[i + 1] if i + 1 < n else ""

        if mode == "line_comment":
            if c == "\n":
                mode = "code"

        elif mode == "block_comment":
            if c == "*" and nxt == "/":
                mode = "code"
                i += 1
                col += 1

        elif mode == "string":
            if c == "\\":
                i += 1
                col += 1
            elif c == string_quote_char:
                mode = "code"
                string_quote_char = None
            elif c == "\n" and not config.multiline_strings_allowed:
                return [{
                    "line_start": line, "line_end": line,
                    "rule": "unterminated_string",
                    "error": "Unterminated string literal",
                    "bug_type": "Syntax Error",
                    "current_code": _line_text(lines, line),
                    "cause": "A quoted string starts on this line but the line ends before a matching "
                             "closing quote is found. Usually a missing closing quote, or a stray quote "
                             "character earlier in the line that was meant to be escaped.",
                }]

        elif mode == "template":
            if config.template_escapes and c == "\\":
                i += 1
                col += 1
            elif c == config.template_quote:
                mode = "code"
            elif config.template_has_interpolation and c == "$" and nxt == "{":
                stack.append({"char": "{", "line": line, "col": col, "is_template_expr": True})
                mode = "code"
                i += 1
                col += 1

        else:  # mode == "code"
            if c == "/" and nxt == "/":
                mode = "line_comment"
                i += 1
                col += 1
            elif c == "/" and nxt == "*":
                mode = "block_comment"
                i += 1
                col += 1
            elif config.template_quote and c == config.template_quote:
                mode = "template"
            elif c in config.string_quotes:
                mode = "string"
                string_quote_char = c
            elif c in BRACKET_OPENERS:
                stack.append({"char": c, "line": line, "col": col, "is_template_expr": False})
            elif c in BRACKET_CLOSERS:
                if not stack:
                    return [{
                        "line_start": line, "line_end": line,
                        "rule": "unexpected_closing_bracket",
                        "error": f"Unexpected '{c}'",
                        "bug_type": "Syntax Error",
                        "current_code": _line_text(lines, line),
                        "cause": f"A closing '{c}' appears here with no matching open bracket anywhere "
                                 f"before it in the file. This is exactly the class of error a bundler "
                                 f"reports as 'Unexpected token' and will fail the build.",
                    }]
                top = stack[-1]
                if top["is_template_expr"] and c == "}":
                    stack.pop()
                    mode = "template"
                elif MATCH[top["char"]] == c:
                    stack.pop()
                else:
                    return [{
                        "line_start": line, "line_end": line,
                        "rule": "mismatched_bracket",
                        "error": f"Mismatched bracket: found '{c}', expected '{MATCH[top['char']]}'",
                        "bug_type": "Syntax Error",
                        "current_code": _line_text(lines, line),
                        "cause": f"'{top['char']}' was opened on line {top['line']} and should be closed "
                                 f"with '{MATCH[top['char']]}', but '{c}' appears instead.",
                    }]

        if c == "\n":
            line += 1
            col = 1
        else:
            col += 1
        i += 1

    if stack:
        first_open = stack[0]
        return [{
            "line_start": first_open["line"], "line_end": first_open["line"],
            "rule": "unclosed_bracket",
            "error": f"Unclosed '{first_open['char']}'",
            "bug_type": "Syntax Error",
            "current_code": _line_text(lines, first_open["line"]),
            "cause": f"This '{first_open['char']}' is opened here but never closed before the end of the "
                     f"file ({len(stack)} bracket{'s' if len(stack) != 1 else ''} left open in total). "
                     f"The build will fail with an 'unexpected end of file' / 'unexpected token' error.",
        }]

    if mode == "string":
        return [{
            "line_start": line, "line_end": line,
            "rule": "unterminated_string",
            "error": "Unterminated string literal at end of file",
            "bug_type": "Syntax Error",
            "current_code": _line_text(lines, line),
            "cause": "A string literal is opened but never closed before the file ends.",
        }]

    if mode == "template":
        return [{
            "line_start": line, "line_end": line,
            "rule": "unterminated_template_literal",
            "error": "Unterminated template literal at end of file",
            "bug_type": "Syntax Error",
            "current_code": _line_text(lines, line),
            "cause": "A template literal (backtick string) is opened but never closed before the file ends.",
        }]

    if mode == "block_comment":
        return [{
            "line_start": line, "line_end": line,
            "rule": "unterminated_comment",
            "error": "Unterminated block comment",
            "bug_type": "Syntax Error",
            "current_code": _line_text(lines, line),
            "cause": "A '/*' comment is opened but '*/' is never found, so everything after it is "
                     "silently commented out until the end of the file — this usually swallows real code.",
        }]

    return []


def mask_non_code(source: str, config: LangConfig = JS_CONFIG) -> str:
    """
    Returns a SAME-LENGTH copy of source with every character inside a
    comment or a string/template literal replaced by a space (newlines
    kept as-is, so line numbers of any match in the result still line up
    with the original file). Used so that regex-based heuristic checks
    (==, var, console.log, etc.) never fire on text that only exists
    inside a comment or a string.
    """
    out = list(source)
    mode = "code"
    string_quote_char = None
    i, n = 0, len(source)

    while i < n:
        c = source[i]
        nxt = source[i + 1] if i + 1 < n else ""

        if mode == "line_comment":
            if c == "\n":
                mode = "code"
            else:
                out[i] = " "

        elif mode == "block_comment":
            if c != "\n":
                out[i] = " "
            if c == "*" and nxt == "/":
                if nxt != "\n":
                    out[i + 1] = " "
                mode = "code"
                i += 1

        elif mode == "string":
            if c != "\n":
                out[i] = " "
            if c == "\\":
                i += 1
                if i < n and source[i] != "\n":
                    out[i] = " "
            elif c == string_quote_char:
                mode = "code"

        elif mode == "template":
            if c != "\n" and c != config.template_quote:
                out[i] = " "
            if config.template_escapes and c == "\\":
                i += 1
                if i < n and source[i] != "\n":
                    out[i] = " "
            elif c == config.template_quote:
                mode = "code"

        else:  # code
            if c == "/" and nxt == "/":
                mode = "line_comment"
                out[i] = " "
            elif c == "/" and nxt == "*":
                mode = "block_comment"
                out[i] = " "
            elif config.template_quote and c == config.template_quote:
                mode = "template"
                out[i] = " "
            elif c in config.string_quotes:
                mode = "string"
                string_quote_char = c
                out[i] = " "

        i += 1

    return "".join(out)
