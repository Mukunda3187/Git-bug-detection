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
    # Set True right when a closing tag's </ is seen, cleared at its own '>'.
    # Needed because </> (a Fragment close) ends with the same '/','>' pair
    # that a self-closing tag like <Home /> ends with - without this flag,
    # the '>' looks like a self-close and decrements depth a second time.
    in_closing_tag = False

    # Tracks the last meaningful (non-whitespace) character seen in code mode,
    # used to guess whether a '/' starts a regex literal or is division -
    # this is the same ambiguity every JS tokenizer has to resolve.
    prev_significant_char = ""
    REGEX_START_TRIGGERS = set("([{,;:=&|!?+-*%<>~^\n")

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
                    "jsx_tag_depth_before": jsx_tag_depth,
                    "in_jsx_text_before": in_jsx_text,
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
                        in_closing_tag = True
                        jsx_tag_depth = max(0, jsx_tag_depth - 1)
                    elif nxt.isalpha() or nxt == ">":
                        # nxt == ">" is a React Fragment <> - it has no tag
                        # name but still opens a JSX scope, exactly like a
                        # named tag does; must be counted or its matching
                        # </> later decrements a level that was never opened.
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

            elif (
                c == "/"
                and nxt not in ("/", "*", ">")
                and (i == 0 or source[i - 1] != "<")
                and (not prev_significant_char or prev_significant_char in REGEX_START_TRIGGERS)
            ):
                # nxt == ">" excluded above: that's a self-closing JSX tag's "/>",
                # not a regex. A newline is a REGEX_START_TRIGGER (for `return\n  /re/`
                # style code), but that means it also precedes "/>" on every
                # multi-line self-closing tag (Prettier's default formatting for
                # any element with multiple attributes) - without this exclusion,
                # that "/" gets treated as a regex start, the scan swallows the
                # ">" while searching for a closing "/", and the tag's own close
                # is silently lost - jsx_tag_depth then never comes back down.
                # Looks like a regex literal (e.g. /^```json\s*/i), not division -
                # skip over it entirely so backticks/quotes INSIDE the pattern
                # (common when stripping markdown fences) don't get mistaken
                # for the start of a string or template literal.
                j = i + 1
                in_char_class = False
                while j < n:
                    cj = source[j]
                    if cj == "\\":
                        j += 2
                        continue
                    if cj == "[":
                        in_char_class = True
                    elif cj == "]":
                        in_char_class = False
                    elif cj == "/" and not in_char_class:
                        j += 1
                        break
                    elif cj == "\n":
                        break  # unterminated regex on this line - bail out, don't skip past it
                    j += 1
                while j < n and source[j].isalpha():
                    j += 1  # trailing flags like /i, /g, /gi

                skipped = source[i:j]
                newlines = skipped.count("\n")
                if newlines:
                    line += newlines
                    col = len(skipped) - skipped.rfind("\n")
                else:
                    col += len(skipped)
                i = j
                prev_significant_char = "/"
                continue

            elif config.template_quote and c == config.template_quote:
                mode = "template"

            elif c in config.string_quotes:
                mode = "string"
                string_quote_char = c

            elif c == "<" and nxt == "/" and not (
                i > 0 and (source[i - 1].isalnum() or source[i - 1] in "_$")
            ):
                # A closing tag like </Routes> can be reached here (rather than
                # via the in_jsx_text branch above) right after a self-closing
                # sibling tag - still needs to decrement, or depth drifts
                # upward forever and real brackets later get swallowed as
                # if they were JSX text. Excludes '<' glued directly onto an
                # identifier (a</b, vanishingly rare) for the same reason as
                # the generic check below.
                in_closing_tag = True
                jsx_tag_depth = max(0, jsx_tag_depth - 1)

            elif c == "<" and (nxt.isalpha() or nxt == ">") and not (
                i > 0 and (source[i - 1].isalnum() or source[i - 1] in "_$")
            ):
                # Enter JSX tag - nxt == ">" is a React Fragment <>, which
                # has no name but still opens a scope that </> will later
                # close, so it must be counted the same as a named tag.
                #
                # The exclusion is the real fix here: a TypeScript generic
                # (Map<string, X>, Record<K, V>, React.FC<Props>, useState<T>())
                # also matches "'<' followed by a letter" - the only reliable
                # difference is that a generic's '<' is glued directly onto an
                # identifier with no space, while a real JSX tag's '<' never is
                # (it's preceded by whitespace, '(', '{', '>', '&&', '?', ':',
                # or start-of-line). Without this, every generic in a .ts/.tsx
                # file falsely increments jsx_tag_depth, and everything after
                # it in the file gets misread as JSX text until things happen
                # to re-sync - or don't, and a real bracket 800 lines later
                # gets reported as the "cause".
                jsx_tag_depth += 1

            elif c == ">" and jsx_tag_depth > 0:
                # If we're currently inside a bracket that was opened while
                # already inside JSX (e.g. the { of element={<Home />}),
                # we're really still inside a JS expression, not JSX text -
                # don't flip modes here even though a nested tag just
                # opened/closed. The matching '}' close already restores the
                # correct in_jsx_text when it's reached.
                inside_jsx_expression = bool(stack) and stack[-1]["jsx_tag_depth_before"] > 0

                if in_closing_tag:
                    # This '>' completes a closing tag (</Foo> or </>) whose
                    # depth was already decremented when its '<' was seen -
                    # do NOT decrement again, even though a Fragment close
                    # </> ends in the same '/','>' pair a self-close does.
                    in_closing_tag = False
                    if not inside_jsx_expression:
                        in_jsx_text = jsx_tag_depth > 0
                elif i > 0 and source[i - 1] == "/":
                    # Self-closing tag like <Home /> opens and closes in the
                    # same breath - it must decrement the depth it just
                    # incremented, and correctly drop back to text mode if a
                    # parent tag is still open, or later characters get
                    # misread.
                    jsx_tag_depth = max(0, jsx_tag_depth - 1)
                    if not inside_jsx_expression:
                        in_jsx_text = jsx_tag_depth > 0
                else:
                    if not inside_jsx_expression:
                        in_jsx_text = True

            elif c in BRACKET_OPENERS:
                stack.append({
                    "char": c,
                    "line": line,
                    "col": col,
                    "is_template_expr": False,
                    "jsx_tag_depth_before": jsx_tag_depth,
                    "in_jsx_text_before": in_jsx_text,
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
                    # A JSX element can appear inside a template interpolation
                    # (e.g. `${<Home />}`) just like inside a JSX attribute
                    # expression - restore whatever JSX state existed right
                    # before this ${ was opened, discarding any drift from
                    # JSX elements used inside it.
                    jsx_tag_depth = top.get("jsx_tag_depth_before", jsx_tag_depth)
                    in_jsx_text = top.get("in_jsx_text_before", in_jsx_text)

                elif MATCH[top["char"]] == c:
                    stack.pop()
                    if c == "}":
                        # A `{...}` JSX expression container (e.g.
                        # element={<Home />}) may contain nested JSX elements
                        # that change jsx_tag_depth/in_jsx_text internally -
                        # once the expression closes, restore exactly the
                        # state that existed right before it opened so none
                        # of that internal JSX drift leaks into the code that
                        # follows.
                        jsx_tag_depth = top.get("jsx_tag_depth_before", jsx_tag_depth)
                        in_jsx_text = top.get("in_jsx_text_before", in_jsx_text)

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

            if c not in (" ", "\t"):
                prev_significant_char = c

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
                        in_jsx_text = False
                        in_closing_tag = True
                        jsx_tag_depth = max(0, jsx_tag_depth - 1)
                    elif nxt.isalpha() or nxt == ">":
                        # nxt == ">" is a React Fragment <> - it has no tag
                        # name but still opens a JSX scope, exactly like a
                        # named tag does; must be counted or its matching
                        # </> later decrements a level that was never opened.
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

            elif (
                c == "/"
                and nxt not in ("/", "*", ">")
                and (i == 0 or source[i - 1] != "<")
                and (not prev_significant_char or prev_significant_char in REGEX_START_TRIGGERS)
            ):
                # nxt == ">" excluded above: that's a self-closing JSX tag's "/>",
                # not a regex. A newline is a REGEX_START_TRIGGER (for `return\n  /re/`
                # style code), but that means it also precedes "/>" on every
                # multi-line self-closing tag (Prettier's default formatting for
                # any element with multiple attributes) - without this exclusion,
                # that "/" gets treated as a regex start, the scan swallows the
                # ">" while searching for a closing "/", and the tag's own close
                # is silently lost - jsx_tag_depth then never comes back down.
                # Looks like a regex literal (e.g. /^```json\s*/i), not division -
                # skip over it entirely so backticks/quotes INSIDE the pattern
                # (common when stripping markdown fences) don't get mistaken
                # for the start of a string or template literal.
                j = i + 1
                in_char_class = False
                while j < n:
                    cj = source[j]
                    if cj == "\\":
                        j += 2
                        continue
                    if cj == "[":
                        in_char_class = True
                    elif cj == "]":
                        in_char_class = False
                    elif cj == "/" and not in_char_class:
                        j += 1
                        break
                    elif cj == "\n":
                        break  # unterminated regex on this line - bail out, don't skip past it
                    j += 1
                while j < n and source[j].isalpha():
                    j += 1  # trailing flags like /i, /g, /gi

                skipped = source[i:j]
                newlines = skipped.count("\n")
                if newlines:
                    line += newlines
                    col = len(skipped) - skipped.rfind("\n")
                else:
                    col += len(skipped)
                i = j
                prev_significant_char = "/"
                continue

            elif config.template_quote and c == config.template_quote:
                mode = "template"

            elif c in config.string_quotes:
                mode = "string"
                string_quote_char = c

            elif c == "<" and nxt == "/" and not (
                i > 0 and (source[i - 1].isalnum() or source[i - 1] in "_$")
            ):
                # A closing tag like </Routes> can be reached here (rather than
                # via the in_jsx_text branch above) right after a self-closing
                # sibling tag - still needs to decrement, or depth drifts
                # upward forever and real brackets later get swallowed as
                # if they were JSX text. Excludes '<' glued directly onto an
                # identifier (a</b, vanishingly rare) for the same reason as
                # the generic check below.
                in_closing_tag = True
                jsx_tag_depth = max(0, jsx_tag_depth - 1)

            elif c == "<" and (nxt.isalpha() or nxt == ">") and not (
                i > 0 and (source[i - 1].isalnum() or source[i - 1] in "_$")
            ):
                # Enter JSX tag - nxt == ">" is a React Fragment <>, which
                # has no name but still opens a scope that </> will later
                # close, so it must be counted the same as a named tag.
                #
                # The exclusion is the real fix here: a TypeScript generic
                # (Map<string, X>, Record<K, V>, React.FC<Props>, useState<T>())
                # also matches "'<' followed by a letter" - the only reliable
                # difference is that a generic's '<' is glued directly onto an
                # identifier with no space, while a real JSX tag's '<' never is
                # (it's preceded by whitespace, '(', '{', '>', '&&', '?', ':',
                # or start-of-line). Without this, every generic in a .ts/.tsx
                # file falsely increments jsx_tag_depth, and everything after
                # it in the file gets misread as JSX text until things happen
                # to re-sync - or don't, and a real bracket 800 lines later
                # gets reported as the "cause".
                jsx_tag_depth += 1

            elif c == ">" and jsx_tag_depth > 0:
                # If we're currently inside a bracket that was opened while
                # already inside JSX (e.g. the { of element={<Home />}),
                # we're really still inside a JS expression, not JSX text -
                # don't flip modes here even though a nested tag just
                # opened/closed. The matching '}' close already restores the
                # correct in_jsx_text when it's reached.
                inside_jsx_expression = bool(stack) and stack[-1]["jsx_tag_depth_before"] > 0

                if in_closing_tag:
                    # This '>' completes a closing tag (</Foo> or </>) whose
                    # depth was already decremented when its '<' was seen -
                    # do NOT decrement again, even though a Fragment close
                    # </> ends in the same '/','>' pair a self-close does.
                    in_closing_tag = False
                    if not inside_jsx_expression:
                        in_jsx_text = jsx_tag_depth > 0
                elif i > 0 and source[i - 1] == "/":
                    # Self-closing tag like <Home /> opens and closes in the
                    # same breath - it must decrement the depth it just
                    # incremented, and correctly drop back to text mode if a
                    # parent tag is still open, or later characters get
                    # misread.
                    jsx_tag_depth = max(0, jsx_tag_depth - 1)
                    if not inside_jsx_expression:
                        in_jsx_text = jsx_tag_depth > 0
                else:
                    if not inside_jsx_expression:
                        in_jsx_text = True

            elif c in BRACKET_OPENERS:
                stack.append({
                    "char": c,
                    "line": line,
                    "col": col,
                    "is_template_expr": False,
                    "jsx_tag_depth_before": jsx_tag_depth,
                    "in_jsx_text_before": in_jsx_text,
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
                    # A JSX element can appear inside a template interpolation
                    # (e.g. `${<Home />}`) just like inside a JSX attribute
                    # expression - restore whatever JSX state existed right
                    # before this ${ was opened, discarding any drift from
                    # JSX elements used inside it.
                    jsx_tag_depth = top.get("jsx_tag_depth_before", jsx_tag_depth)
                    in_jsx_text = top.get("in_jsx_text_before", in_jsx_text)

                elif MATCH[top["char"]] == c:
                    stack.pop()
                    if c == "}":
                        # A `{...}` JSX expression container (e.g.
                        # element={<Home />}) may contain nested JSX elements
                        # that change jsx_tag_depth/in_jsx_text internally -
                        # once the expression closes, restore exactly the
                        # state that existed right before it opened so none
                        # of that internal JSX drift leaks into the code that
                        # follows.
                        jsx_tag_depth = top.get("jsx_tag_depth_before", jsx_tag_depth)
                        in_jsx_text = top.get("in_jsx_text_before", in_jsx_text)

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

            if c not in (" ", "\t"):
                prev_significant_char = c

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
                stack.append({
                    "char": "{",
                    "line": line,
                    "col": col,
                    "is_template_expr": True,
                    "jsx_tag_depth_before": jsx_tag_depth,
                    "in_jsx_text_before": in_jsx_text,
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
                        in_closing_tag = True
                        jsx_tag_depth = max(0, jsx_tag_depth - 1)
                    elif nxt.isalpha() or nxt == ">":
                        # nxt == ">" is a React Fragment <> - it has no tag
                        # name but still opens a JSX scope, exactly like a
                        # named tag does; must be counted or its matching
                        # </> later decrements a level that was never opened.
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

            elif (
                c == "/"
                and nxt not in ("/", "*")
                and (i == 0 or source[i - 1] != "<")
                and (not prev_significant_char or prev_significant_char in REGEX_START_TRIGGERS)
            ):
                # Looks like a regex literal (e.g. /^```json\s*/i), not division -
                # skip over it entirely so backticks/quotes INSIDE the pattern
                # (common when stripping markdown fences) don't get mistaken
                # for the start of a string or template literal.
                j = i + 1
                in_char_class = False
                while j < n:
                    cj = source[j]
                    if cj == "\\":
                        j += 2
                        continue
                    if cj == "[":
                        in_char_class = True
                    elif cj == "]":
                        in_char_class = False
                    elif cj == "/" and not in_char_class:
                        j += 1
                        break
                    elif cj == "\n":
                        break  # unterminated regex on this line - bail out, don't skip past it
                    j += 1
                while j < n and source[j].isalpha():
                    j += 1  # trailing flags like /i, /g, /gi

                skipped = source[i:j]
                newlines = skipped.count("\n")
                if newlines:
                    line += newlines
                    col = len(skipped) - skipped.rfind("\n")
                else:
                    col += len(skipped)
                i = j
                prev_significant_char = "/"
                continue

            elif config.template_quote and c == config.template_quote:
                mode = "template"

            elif c in config.string_quotes:
                mode = "string"
                string_quote_char = c

            elif c == "<" and nxt == "/":
                # A closing tag like </Routes> can be reached here (rather than
                # via the in_jsx_text branch above) right after a self-closing
                # sibling tag - still needs to decrement, or depth drifts
                # upward forever and real brackets later get swallowed as
                # if they were JSX text.
                in_closing_tag = True
                jsx_tag_depth = max(0, jsx_tag_depth - 1)

            elif c == "<" and (nxt.isalpha() or nxt == ">"):
                # Enter JSX tag - nxt == ">" is a React Fragment <>, which
                # has no name but still opens a scope that </> will later
                # close, so it must be counted the same as a named tag.
                jsx_tag_depth += 1

            elif c == ">" and jsx_tag_depth > 0:
                # If we're currently inside a bracket that was opened while
                # already inside JSX (e.g. the { of element={<Home />}),
                # we're really still inside a JS expression, not JSX text -
                # don't flip modes here even though a nested tag just
                # opened/closed. The matching '}' close already restores the
                # correct in_jsx_text when it's reached.
                inside_jsx_expression = bool(stack) and stack[-1]["jsx_tag_depth_before"] > 0

                if in_closing_tag:
                    # This '>' completes a closing tag (</Foo> or </>) whose
                    # depth was already decremented when its '<' was seen -
                    # do NOT decrement again, even though a Fragment close
                    # </> ends in the same '/','>' pair a self-close does.
                    in_closing_tag = False
                    if not inside_jsx_expression:
                        in_jsx_text = jsx_tag_depth > 0
                elif i > 0 and source[i - 1] == "/":
                    # Self-closing tag like <Home /> opens and closes in the
                    # same breath - it must decrement the depth it just
                    # incremented, and correctly drop back to text mode if a
                    # parent tag is still open, or later characters get
                    # misread.
                    jsx_tag_depth = max(0, jsx_tag_depth - 1)
                    if not inside_jsx_expression:
                        in_jsx_text = jsx_tag_depth > 0
                else:
                    if not inside_jsx_expression:
                        in_jsx_text = True

            elif c in BRACKET_OPENERS:
                stack.append({
                    "char": c,
                    "line": line,
                    "col": col,
                    "is_template_expr": False,
                    "jsx_tag_depth_before": jsx_tag_depth,
                    "in_jsx_text_before": in_jsx_text,
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
                    # A JSX element can appear inside a template interpolation
                    # (e.g. `${<Home />}`) just like inside a JSX attribute
                    # expression - restore whatever JSX state existed right
                    # before this ${ was opened, discarding any drift from
                    # JSX elements used inside it.
                    jsx_tag_depth = top.get("jsx_tag_depth_before", jsx_tag_depth)
                    in_jsx_text = top.get("in_jsx_text_before", in_jsx_text)

                elif MATCH[top["char"]] == c:
                    stack.pop()
                    if c == "}":
                        # A `{...}` JSX expression container (e.g.
                        # element={<Home />}) may contain nested JSX elements
                        # that change jsx_tag_depth/in_jsx_text internally -
                        # once the expression closes, restore exactly the
                        # state that existed right before it opened so none
                        # of that internal JSX drift leaks into the code that
                        # follows.
                        jsx_tag_depth = top.get("jsx_tag_depth_before", jsx_tag_depth)
                        in_jsx_text = top.get("in_jsx_text_before", in_jsx_text)

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

            if c not in (" ", "\t"):
                prev_significant_char = c

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
