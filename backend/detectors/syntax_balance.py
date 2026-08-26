from dataclasses import dataclass


@dataclass
class LangConfig:
    string_quotes: str = "'\""
    template_quote: str = None
    template_has_interpolation: bool = True
    template_escapes: bool = True
    multiline_strings_allowed: bool = False


JS_CONFIG = LangConfig(
    template_quote="`",
    template_has_interpolation=True,
    template_escapes=True,
)

GO_CONFIG = LangConfig(
    template_quote="`",
    template_has_interpolation=False,
    template_escapes=False,
)

C_FAMILY_CONFIG = LangConfig(template_quote=None)

BRACKET_OPENERS = "([{"
BRACKET_CLOSERS = ")]}"
MATCH = {"(": ")", "[": "]", "{": "}"}


def _line_text(lines, ln):
    return lines[ln - 1].strip() if 0 < ln <= len(lines) else ""


def check_bracket_balance(source: str, config: LangConfig = JS_CONFIG):
    lines = source.splitlines()
    stack = []

    mode = "code"
    string_quote_char = None
    line, col = 1, 1
    i, n = 0, len(source)

    # JSX state
    jsx_tag_depth = 0
    in_jsx_text = False

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
                    "line_start": line,
                    "line_end": line,
                    "rule": "unterminated_string",
                    "error": "Unterminated string literal",
                    "bug_type": "Syntax Error",
                    "current_code": _line_text(lines, line),
                    "cause": "A quoted string starts here but does not have a matching closing quote.",
                }]

        elif mode == "template":
            if config.template_escapes and c == "\\":
                i += 1
                col += 1
            elif c == config.template_quote:
                mode = "code"
            elif config.template_has_interpolation and c == "$" and nxt == "{":
                stack.append({
                    "char": "{",
                    "line": line,
                    "col": col,
                    "is_template_expr": True,
                })
                mode = "code"
                i += 1
                col += 1

        else:
            # JSX text: ignore quotes such as the ' in "isn't".
            if in_jsx_text:
                if c == "<":
                    if nxt == "/":
                        in_jsx_text = False
                        jsx_tag_depth = max(0, jsx_tag_depth - 1)
                    elif nxt.isalpha():
                        in_jsx_text = False
                        jsx_tag_depth += 1

                # Brackets in normal JSX text are just text.
                if c == "\n":
                    line += 1
                    col = 1
                else:
                    col += 1

                i += 1
                continue

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

            elif c == "<" and nxt.isalpha():
                # Enter JSX tag.
                jsx_tag_depth += 1

            elif c == ">" and jsx_tag_depth > 0:
                # Check whether this is a normal opening JSX tag.
                if i > 0 and source[i - 1] != "/":
                    in_jsx_text = True

            elif c in BRACKET_OPENERS:
                stack.append({
                    "char": c,
                    "line": line,
                    "col": col,
                    "is_template_expr": False,
                })

            elif c in BRACKET_CLOSERS:
                if not stack:
                    return [{
                        "line_start": line,
                        "line_end": line,
                        "rule": "unexpected_closing_bracket",
                        "error": f"Unexpected '{c}'",
                        "bug_type": "Syntax Error",
                        "current_code": _line_text(lines, line),
                        "cause": f"There is a closing '{c}' here, but there is no matching opening bracket.",
                    }]

                top = stack[-1]

                if top["is_template_expr"] and c == "}":
                    stack.pop()
                    mode = "template"

                elif MATCH[top["char"]] == c:
                    stack.pop()

                else:
                    return [{
                        "line_start": line,
                        "line_end": line,
                        "rule": "mismatched_bracket",
                        "error": f"Mismatched bracket: found '{c}', expected '{MATCH[top['char']]}'",
                        "bug_type": "Syntax Error",
                        "current_code": _line_text(lines, line),
                        "cause": f"The bracket opened on line {top['line']} needs '{MATCH[top['char']]}', but '{c}' was used instead.",
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
            "line_start": first_open["line"],
            "line_end": first_open["line"],
            "rule": "unclosed_bracket",
            "error": f"Unclosed '{first_open['char']}'",
            "bug_type": "Syntax Error",
            "current_code": _line_text(lines, first_open["line"]),
            "cause": f"This '{first_open['char']}' was opened here but was never closed.",
        }]

    if mode == "string":
        return [{
            "line_start": line,
            "line_end": line,
            "rule": "unterminated_string",
            "error": "Unterminated string literal at end of file",
            "bug_type": "Syntax Error",
            "current_code": _line_text(lines, line),
            "cause": "A text string was opened but never closed.",
        }]

    if mode == "template":
        return [{
            "line_start": line,
            "line_end": line,
            "rule": "unterminated_template_literal",
            "error": "Unterminated template literal at end of file",
            "bug_type": "Syntax Error",
            "current_code": _line_text(lines, line),
            "cause": "A template string was opened but never closed.",
        }]

    if mode == "block_comment":
        return [{
            "line_start": line,
            "line_end": line,
            "rule": "unterminated_comment",
            "error": "Unterminated block comment",
            "bug_type": "Syntax Error",
            "current_code": _line_text(lines, line),
            "cause": "A comment was opened with /* but was never closed with */.",
        }]

    return []


def mask_non_code(source: str, config: LangConfig = JS_CONFIG) -> str:
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

        else:
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
