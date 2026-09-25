"""
Wraps the call to the LLM (Google Gemini). Give it:
  - a raw candidate finding from a detector (detectors/python_detector.py)
  - the retrieved similar historical bugs from the RAG step
and it returns a fully-formed structured bug report as a dict, matching
models.BugReport (minus the fields the caller already knows, like id/file).

Uses Gemini's REST API directly via `requests` (already a dependency) -
no extra SDK to install or version-pin.

If no GEMINI_API_KEY is set, falls back to a transparent rule-based
formatter so the app still runs end-to-end for a demo - it clearly does
NOT pretend to be the LLM's reasoning, it just formats what the detector
already found.
"""
import json
import math
import os
import re

import requests

GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

# See the comment in analyze_finding() where this is used - together these two
# bound the worst-case time one finding's LLM call can take to
# MAX_KEYS_TO_TRY_PER_CALL * GEMINI_REQUEST_TIMEOUT_SECONDS, regardless of how
# many keys end up configured.
MAX_KEYS_TO_TRY_PER_CALL = 4
GEMINI_REQUEST_TIMEOUT_SECONDS = 7


def _parse_rate_limit_info(resp):
    """
    Reads Google's 429 error body to figure out how long to wait and whether
    this is a short per-minute limit or the daily free-tier cap. Returns
    (retry_seconds_or_None, is_daily_limit).
    """
    try:
        data = resp.json()
    except (ValueError, json.JSONDecodeError):
        return None, False

    error = data.get("error", {})
    details = error.get("details", [])
    retry_seconds = None
    is_daily = False

    for d in details:
        type_str = d.get("@type", "")
        if type_str.endswith("RetryInfo"):
            delay = d.get("retryDelay", "")
            try:
                retry_seconds = float(str(delay).rstrip("s"))
            except ValueError:
                pass
        if type_str.endswith("QuotaFailure"):
            for v in d.get("violations", []):
                combined = f"{v.get('quotaId', '')} {v.get('quotaMetric', '')}"
                if "PerDay" in combined or "per_day" in combined:
                    is_daily = True

    return retry_seconds, is_daily


def _build_rate_limit_message(retry_seconds, is_daily):
    """Turns the parsed rate-limit details into one plain-English sentence."""
    if is_daily:
        if retry_seconds:
            minutes = max(1, math.ceil(retry_seconds / 60))
            return f"The free AI usage limit for today has been reached. Try again in about {minutes} minute(s), or after Google's daily reset (midnight Pacific Time, US)."
        return "The free AI usage limit for today has been reached. It resets at midnight Pacific Time (US) - please try again after that."

    if retry_seconds:
        if retry_seconds < 60:
            wait = int(retry_seconds) + 5  # small buffer
            return f"The AI is temporarily busy. Try again in about {wait} seconds."
        minutes = max(1, math.ceil(retry_seconds / 60))
        return f"The AI is temporarily busy. Try again in about {minutes} minute(s)."

    return "The AI usage limit has been reached for now. Please try again in a minute or two."

SYSTEM_PROMPT = """You are an expert code reviewer. Give an immediately usable fix for the exact bug shown.

For each issue:
1. cause: Explain why this exact code is wrong in 1-2 short sentences.
2. solution_type: Choose "replace", "add", "remove", or "create_file".
3. replacement_code:
   - For "replace", give the COMPLETE corrected replacement for the supplied current_code.
   - It must be directly copy-pasteable.
   - Change only what is necessary to fix the bug.
   - For "add", give the exact code that must be added when possible.
   - For "remove", give the exact code that should be removed when possible.
4. solution: EXACTLY 2 short sentences:
   Sentence 1: what the developer must do.
   Sentence 2: why that fixes this bug.
   Do NOT give theory, background, long explanations, or generic advice.
5. explanation: one short technical sentence.
6. confidence: integer 0-100 for this specific diagnosis and fix.
7. insufficient_evidence: true if the supplied code is not enough to create a reliable exact fix.

CRITICAL:
- The replacement_code is the actual fix shown to the user.
- Never invent code when the supplied snippet is insufficient.
- If an exact replacement cannot be determined safely, set insufficient_evidence=true.
- Do not put markdown fences around replacement_code.
- Do not include code inside solution; code belongs in replacement_code.
- Keep solution to exactly two short sentences.

Reply with ONLY a JSON object:
{
  "error": string,
  "bug_type": one of ["Runtime Error","Logic Error","Syntax Error","Type Error","Dependency Error","Security Issue","Performance Issue","API Error","Unnecessary Code","Other"],
  "cause": string,
  "why_occurs": string,
  "solution_type": one of ["replace","add","remove","create_file"],
  "solution": string,
  "replacement_code": string or null,
  "add_location": string or null,
  "new_file_path": string or null,
  "explanation": string,
  "confidence": integer from 0 to 100,
  "insufficient_evidence": boolean
}"""


def _build_user_message(finding: dict, retrieved: list) -> str:
    retrieved_block = "\n\n".join(
        f"- Dataset: {r['record'].get('dataset_source')}\n"
        f"  Bug type: {r['record'].get('bug_type')}\n"
        f"  Description: {r['record'].get('bug_description')}\n"
        f"  Solution: {r['record'].get('solution', 'N/A')}\n"
        f"  Similarity: {round(r['similarity'] * 100)}%"
        for r in retrieved
    ) or "(no similar historical bugs found)"

    return f"""**Candidate Issue from Static Analysis:**

**File:** {finding.get('file')}
**Function:** {finding.get('function') or 'N/A'}
**Lines:** {finding.get('line_start')}-{finding.get('line_end')}
**Rule:** {finding.get('rule')}
**Initial Assessment:** {finding.get('error')} ({finding.get('bug_type')})
**Detector's Note:** {finding.get('cause')}

**Current Code:**
```
{finding.get('current_code')}
```

**Historical Context (Similar Bugs from Knowledge Base):**
{retrieved_block}

**Your Task:**
Analyze this candidate and provide the smallest reliable code replacement that fixes the reported bug. The replacement_code must be directly usable for the supplied Current Code. If the exact fix cannot be determined from the supplied code, set insufficient_evidence=true instead of inventing code."""


# How confident the FALLBACK path (no live LLM call) can honestly be in
# its own suggested fix, per rule. This varies deliberately - a rule with
# an exact, verified string-substitution fix (like eq_none: "== None" -> "is None")
# deserves a much higher number than one with only generic advice and no
# computed replacement (like empty_catch_block, which just says "add a
# console.error"). This is about confidence in the FIX, not confidence
# that the underlying finding is real - every finding these detectors
# produce is already a 100%-certain fact (a real syntax error, or
# provably unreachable code), independent of this fallback quality score.
FALLBACK_CONFIDENCE_BY_RULE = {
    "possibly_unused_function": 60,
    "bare_except": 85,
    "mutable_default_arg": 55,
    "eq_none": 90,
    "possible_division_by_zero": 50,
    "syntax_error": 70,
    "unclosed_bracket": 65,
    "mismatched_bracket": 65,
    "unexpected_closing_bracket": 65,
    "unterminated_string": 75,
    "unterminated_template_literal": 75,
    "unterminated_comment": 75,
    "loose_equality": 90,
    "var_declaration": 85,
    "empty_catch_block": 55,
    "leftover_console_statement": 90,
    "leftover_debugger_statement": 90,
    "leftover_debug_print": 85,
    "unreachable_code": 95,  # deleting provably-dead code is always a safe, correct fix
}
DEFAULT_FALLBACK_CONFIDENCE = 40

# Pulls the exact character(s) the detector already identified out of its
# "error" string, so the fallback solution can name them specifically
# ("add the missing '}'") instead of falling back to generic advice
# ("check every bracket near this line") when the detector already knows
# precisely which bracket and which problem this is.
UNCLOSED_CHAR_RE = re.compile(r"Unclosed '(.)'")
MISMATCHED_CHARS_RE = re.compile(r"found '(.)', expected '(.)'")
UNEXPECTED_CHAR_RE = re.compile(r"Unexpected '(.)'")
BRACKET_CLOSER = {"(": ")", "[": "]", "{": "}"}

# Matches a single-line-signature mutable default like "bucket=[]", "cfg={}",
# or "seen=set()" - covers the common case the fallback path can fix safely
# without a real parser (multi-line signatures are left alone, see below).
MUTABLE_DEFAULT_RE = re.compile(r"(\w+)\s*=\s*(\[\]|\{\}|set\(\))")

# Matches an empty catch block, with or without a captured error variable,
# e.g. "catch (err) {}", "catch (Exception e) {}", or "catch {}".
EMPTY_CATCH_WITH_PARAM_RE = re.compile(r"catch\s*\(([^)]*)\)\s*\{\s*\}")
EMPTY_CATCH_NO_PARAM_RE = re.compile(r"catch\s*\{\s*\}")

CATCH_LOG_STATEMENT_BY_EXTENSION = {
    ".js": "console.error({var});", ".jsx": "console.error({var});",
    ".ts": "console.error({var});", ".tsx": "console.error({var});",
    ".java": "{var}.printStackTrace();",
    ".cs": "Console.WriteLine({var});",
    ".cpp": "std::cerr << {var}.what() << std::endl;",
    ".php": "error_log({var}->getMessage());",
}

# Matches a simple denominator right after a '/': a plain identifier,
# optionally followed by attribute access (.name) or indexing ([key]).
# Deliberately does NOT match a parenthesized expression, a function call,
# or an arithmetic expression on the right of the '/' - for "x / (a + b)" or
# "x / get_count()" there's no single safe variable name to point at, so
# _extract_division_denominator returns None for those rather than guessing.
DIVISION_DENOMINATOR_RE = re.compile(r"/\s*([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*|\[[^\]\[]+\])*)")

PYTHON_MISSING_COLON_RE = re.compile(r"expected ':'")
PYTHON_UNCLOSED_BRACKET_RE = re.compile(r"'(.)' was never closed")


def _extract_division_denominator(current_code: str):
    """Pulls the real denominator out of a simple division expression
    (e.g. "total / count" -> "count", "x / self.n" -> "self.n"), so the
    suggested zero-check names the actual variable instead of a generic
    placeholder every division-by-zero finding used to show identically."""
    if not current_code:
        return None
    m = DIVISION_DENOMINATOR_RE.search(current_code)
    return m.group(1) if m else None


def _fix_unterminated_string_line(current_code: str):
    """Appends whichever quote character has an odd count on this line - the
    one the detector already identified as never closed. Works for both the
    JS/TS structural check and Python's own "unterminated string literal"
    parser message, since both report the single offending line as
    current_code. Returns None if neither quote type has an odd count (not
    expected for a genuinely unterminated string, but this is a heuristic
    over plain text, not a real parser, so it stays conservative rather than
    guessing)."""
    if not current_code or not current_code.strip():
        return None
    if current_code.count('"') % 2 == 1:
        return current_code.rstrip() + '"'
    if current_code.count("'") % 2 == 1:
        return current_code.rstrip() + "'"
    return None


def _fix_python_syntax_error(current_code: str, cause_from_detector: str):
    """Return a conservative, copy-pasteable fix for common Python syntax errors.

    This is a fallback used when the LLM is unavailable. It intentionally
    handles only transformations that can be derived from the parser message
    and the supplied snippet; ambiguous semantic repairs are left without a
    fabricated replacement.
    """
    if not current_code or not current_code.strip():
        return None

    code = current_code.rstrip()
    cause = (cause_from_detector or "").lower()

    # A required parameter cannot follow a parameter with a default.
    # Move required simple parameters before defaulted parameters, preserving
    # the default expression. Only handle simple signatures without annotations,
    # *args, /, or **kwargs, where comma splitting is unambiguous.
    if "non-default argument follows default argument" in cause:
        match = re.search(
            r"(?m)^(\s*(?:async\s+)?def\s+\w+\s*\()([^()\n]*)(\)\s*:\s*)$",
            code,
        )
        if match:
            params = [p.strip() for p in match.group(2).split(",")]
            if params and all(
                p and not any(ch in p for ch in (":", "*", "/"))
                for p in params
            ):
                required = [p for p in params if "=" not in p]
                optional = [p for p in params if "=" in p]
                if required and optional:
                    fixed_signature = (
                        match.group(1)
                        + ", ".join(required + optional)
                        + match.group(3)
                    )
                    return code[:match.start()] + fixed_signature + code[match.end():]

    # Missing comma between simple function parameters, e.g. def add(a b):
    if "forgot a comma" in cause or "invalid syntax" in cause:
        def_sig = re.search(
            r"(?m)^(\s*(?:async\s+)?def\s+\w+\s*\()([^()\n]*)(\)\s*:\s*)$",
            code,
        )
        if def_sig and not any(ch in def_sig.group(2) for ch in (":", "*", "/")):
            params = def_sig.group(2).strip()
            fixed = re.sub(
                r"(?<=[A-Za-z0-9_])\s+(?=[A-Za-z_]\w*(?:\s*=|\s*,|\s*$))",
                ", ",
                params,
            )
            if fixed != params:
                fixed_signature = def_sig.group(1) + fixed + def_sig.group(3)
                return code[:def_sig.start()] + fixed_signature + code[def_sig.end():]

        # Common one-line suite typo: `if condition print(...)` -> `if condition: print(...)`.
        for keyword in ("if", "elif", "while"):
            m = re.match(
                rf"^(\s*{keyword}\s+.+?)\s+(?=(?:print|return|raise|pass|break|continue)\b)",
                code,
            )
            if m and ":" not in m.group(1):
                return code[:m.end(1)] + ":" + code[m.end(1):]

    # Python explicitly reports a missing colon for compound statements.
    if "expected ':'" in cause:
        if not code.endswith(":"):
            return code + ":"

    # Missing closing bracket/parenthesis reported by Python.
    m = re.search(r"'(.)' was never closed", cause_from_detector or "")
    if m:
        closer = BRACKET_CLOSER.get(m.group(1))
        if closer:
            return code + closer

    # Unterminated string literal.
    if "unterminated string literal" in cause or "unterminated triple-quoted string literal" in cause:
        return _fix_unterminated_string_line(code)

    # A simple unmatched extra closing bracket on the reported snippet.
    if "unmatched ')'" in cause or "unmatched ']'" in cause or "unmatched '}'" in cause:
        for closer in (")", "]", "}"):
            if closer in code:
                return code[:code.rfind(closer)] + code[code.rfind(closer) + 1:]

    # Incomplete `from module import` can be repaired without guessing a name
    # by importing the module itself.
    if re.fullmatch(r"\s*from\s+([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s+import\s*", code):
        m = re.fullmatch(r"\s*from\s+([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s+import\s*", code)
        if m:
            indent = code[:len(code) - len(code.lstrip())]
            return f"{indent}import {m.group(1)}"

    return None


def _fix_mutable_default_arg(current_code: str):
    """
    Mechanically rewrites a one-line function signature with a mutable
    default argument into the safe None-default pattern, e.g.:
        def add_item(item, bucket=[]):        ->  def add_item(item, bucket=None):
            bucket.append(item)                        if bucket is None:
                                                            bucket = []
                                                        bucket.append(item)
    Returns None (never a half-applied guess) when the default can't be
    confidently located, so the caller can fall back to plain-text advice
    instead of claiming code is shown when it isn't.
    """
    lines = current_code.splitlines()
    if not lines:
        return None

    match = MUTABLE_DEFAULT_RE.search(lines[0])
    if not match:
        return None

    param_name, literal = match.group(1), match.group(2)
    default_value = {"[]": "[]", "{}": "{}", "set()": "set()"}[literal]
    new_signature = lines[0][:match.start()] + f"{param_name}=None" + lines[0][match.end():]

    body_lines = lines[1:]
    indent = "    "
    if body_lines:
        stripped = body_lines[0].lstrip()
        indent = body_lines[0][:len(body_lines[0]) - len(stripped)] or indent

    guard = [f"{indent}if {param_name} is None:", f"{indent}    {param_name} = {default_value}"]
    return "\n".join([new_signature] + guard + body_lines)


def _fix_empty_catch(current_code: str, file_path: str):
    """
    Mechanically rewrites a one-line empty catch block into one that logs
    the error, using a log statement appropriate to the file's language
    (inferred from its extension) and re-using whatever variable name the
    catch clause already captured. Returns None if no empty catch pattern
    is found, rather than guessing.
    """
    ext = os.path.splitext(file_path or "")[1]
    log_template = CATCH_LOG_STATEMENT_BY_EXTENSION.get(ext)
    if not log_template:
        return None

    match = EMPTY_CATCH_WITH_PARAM_RE.search(current_code)
    if match:
        param_str = match.group(1).strip()
        var_match = re.search(r"(\$?\w+)\s*$", param_str)
        var_name = var_match.group(1) if var_match else "e"
        log_stmt = log_template.format(var=var_name)
        return current_code[:match.start()] + f"catch ({param_str}) {{ {log_stmt} }}" + current_code[match.end():]

    match = EMPTY_CATCH_NO_PARAM_RE.search(current_code)
    if match:
        var_name = "err" if ext in (".js", ".jsx", ".ts", ".tsx") else "e"
        log_stmt = log_template.format(var=var_name)
        return current_code[:match.start()] + f"catch ({var_name}) {{ {log_stmt} }}" + current_code[match.end():]

    return None


def _fallback_report(finding: dict) -> dict:
    """Used when no LLM reasoning is available - either no API key is configured, or the
    live LLM call failed/timed out. Kept in plain, non-technical English since this is
    shown directly to the end user - technical failure details are logged to the server
    console instead (see analyze_finding), never shown in the UI.

    Every rule every detector can produce gets a real, specific template here - no rule
    should ever fall through to a generic "review and fix it" message. Each one sets a
    single `solution` string that stands completely on its own (what's wrong here,
    specifically, and exactly what to do about it) rather than the old split between a
    generic intro sentence and a separate action sentence - for the three bracket rules
    in particular, that generic intro used to be the same wording no matter which
    specific bracket or which specific problem the detector had already identified, even
    though that exact detail (which character, unclosed vs mismatched vs unexpected) was
    sitting right there in the detector's own `error` string. This pulls it out and names
    it directly instead of leaving the person to re-derive it from the code themselves.
    """
    rule = finding.get("rule", "")
    bug_type = finding.get("bug_type", "Other")
    current_code = finding.get("current_code", "") or ""
    error_text = finding.get("error", "") or ""
    cause_from_detector = finding.get("cause", "")

    # Defaults - overridden below per rule. solution_type "replace" needs
    # replacement_code; "remove" and "add" generally don't need it filled in
    # for a deterministic fallback since there's nothing left to guess.
    solution_type = "replace"
    solution = ""
    replacement_code = None
    add_location = None
    cause = cause_from_detector

    if rule == "possibly_unused_function":
        solution_type = "remove"
        replacement_code = current_code
        solution = "Remove this unused function if it is not referenced elsewhere. This removes dead code without changing behavior when there are no external references."

    elif rule == "bare_except":
        solution_type = "replace"
        replacement_code = current_code.replace("except:", "except Exception as e:", 1)
        solution = "Replace the bare 'except:' with 'except Exception as e:' using the code below. This catches normal exceptions without also swallowing system-exit and keyboard-interrupt signals."

    elif rule == "mutable_default_arg":
        replacement_code = _fix_mutable_default_arg(current_code)
        if replacement_code:
            solution_type = "replace"
            solution = "Replace the function code with the version below so the mutable value is created separately for each call. This prevents data from one function call being reused by the next call."
        else:
            # Couldn't mechanically locate the mutable default in the captured
            # snippet (e.g. a multi-line signature) - never claim code is
            # "shown below" and then show nothing.
            solution_type = "add"
            solution = "Change the mutable default to None and create a new list, dict, or set inside the function. This prevents the same mutable object from being shared across calls."

    elif rule == "eq_none":
        solution_type = "replace"
        replacement_code = current_code.replace("== None", "is None").replace("!= None", "is not None")
        solution = "Replace the None comparison with 'is None' or 'is not None' using the code below. This performs the intended identity check for Python's None value."

    elif rule == "possible_division_by_zero":
        solution_type = "add"
        denominator = _extract_division_denominator(current_code)
        if denominator:
            add_location = "Add this check on the line right before the division."
            replacement_code = f"if {denominator} != 0:"
            solution = f"Add a check that '{denominator}' isn't zero immediately before this line, using the code below, and handle the zero case explicitly (skip the calculation, use a default, or raise a clear error). This prevents the calculation from raising ZeroDivisionError at runtime."
        else:
            solution = "Add a check that the denominator isn't zero immediately before this division, and handle the zero case explicitly (skip the calculation, use a default value, or raise a clear error). This prevents the calculation from raising ZeroDivisionError at runtime."

    elif rule == "syntax_error":
        cause = f"Python's own parser could not read this code. The exact reason it gave was: \"{cause_from_detector}\"."
        replacement_code = _fix_python_syntax_error(current_code, cause_from_detector)
        if replacement_code:
            solution_type = "replace"
            solution = f"Replace this line with the version below to fix the exact problem Python reported: {cause_from_detector}. Re-run the file afterward to confirm the syntax error is gone."
        else:
            solution_type = "add"
            add_location = "Edit the reported line and correct the parser error."
            solution = f"Fix the parser problem reported here: {cause_from_detector}. Re-run the file after editing to confirm that the syntax error is gone."

    elif rule == "unclosed_bracket":
        solution_type = "replace"
        m = UNCLOSED_CHAR_RE.search(error_text)
        opener = m.group(1) if m else "{"
        closer = BRACKET_CLOSER.get(opener, "}")
        if current_code.strip():
            replacement_code = current_code.rstrip() + closer
        solution = (
            f"Add the missing '{closer}' to close the '{opener}' opened by this code. "
            f"Place it where the block or expression ends so the brackets are balanced."
        )

    elif rule == "mismatched_bracket":
        solution_type = "replace"
        m = MISMATCHED_CHARS_RE.search(error_text)
        found, expected = (m.group(1), m.group(2)) if m else ("?", "?")
        if found in current_code and expected:
            replacement_code = current_code.replace(found, expected, 1)
        solution = (
            f"Replace the unexpected '{found}' with the expected '{expected}' when this is the closing bracket for the current block. "
            f"If the surrounding structure uses '{found}' intentionally, check the earlier opening bracket instead."
        )

    elif rule == "unexpected_closing_bracket":
        m = UNEXPECTED_CHAR_RE.search(error_text)
        closer = m.group(1) if m else "}"
        remainder = current_code.replace(closer, "", 1).strip() if closer in current_code else None
        if remainder:
            # Something else shares this line with the stray bracket (e.g.
            # "});" where only the "}" is extra) - show the corrected line.
            solution_type = "replace"
            replacement_code = current_code.replace(closer, "", 1)
            solution = f"Remove the extra '{closer}' from this line, using the code below. If it's meant to close a real block instead, add the missing opening bracket at that block's start."
        elif remainder == "":
            # The stray bracket is the entire line - "replacing" it would
            # just be an empty code block, which is confusing to show as a
            # "here's your fix" box. Say delete the line instead.
            solution_type = "remove"
            solution = f"Delete this line - the '{closer}' on it has no matching opening bracket anywhere before it in the file. If it's meant to close a real block instead, add the missing opening bracket at that block's start rather than deleting this line."
        else:
            solution_type = "replace"
            solution = f"Remove the extra '{closer}' if it has no matching opening bracket. If it's meant to close a real block, add the missing opening bracket at that block's start instead."

    elif rule == "unterminated_string":
        solution_type = "replace"
        replacement_code = _fix_unterminated_string_line(current_code)
        if replacement_code:
            solution = "Add the missing closing quote using the code below - it matches whichever quote character opened the string."
        else:
            solution = "Add the missing closing quote at the end of this string - it needs to match whichever quote character (' or \") opened it. A string can't span multiple lines unless it's a template literal (backticks) or a triple-quoted string, so a missing quote here usually means the string was meant to end on this same line."

    elif rule == "unterminated_template_literal":
        solution_type = "replace"
        if current_code.strip():
            replacement_code = current_code.rstrip() + "`"
        solution = "Add the missing closing backtick (`) to complete this template literal. This lets the parser treat the following code as code instead of continuing the string."

    elif rule == "unterminated_comment":
        solution_type = "replace"
        if current_code.strip():
            replacement_code = current_code.rstrip() + "*/"
        solution = "Add the missing */ to close this block comment. Then check the following lines to ensure no real code was accidentally included inside the comment."

    elif rule == "loose_equality":
        solution_type = "replace"
        if "!=" in current_code:
            replacement_code = current_code.replace("!=", "!==")
            solution = "Change '!=' to '!==' using the code below. This compares both type and value without implicit type conversion."
        else:
            replacement_code = current_code.replace("==", "===")
            solution = "Change '==' to '===' using the code below. This compares both type and value without implicit type conversion."

    elif rule == "var_declaration":
        solution_type = "replace"
        replacement_code = current_code.replace("var ", "let ", 1)
        solution = "Change 'var' to 'let' or 'const' using the code below. Block-scoped declarations prevent the variable from leaking outside its intended block."

    elif rule == "empty_catch_block":
        replacement_code = _fix_empty_catch(current_code, finding.get("file", ""))
        if replacement_code:
            solution_type = "replace"
            solution = "Replace this empty catch block with the version below so the error is recorded. This prevents failures from being silently discarded."
        else:
            solution_type = "add"
            solution = "Add an error log inside the catch block. This keeps the failure visible instead of silently discarding the exception."

    elif rule == "leftover_console_statement":
        solution_type = "remove"
        replacement_code = current_code
        solution = "Remove this debugging console statement if it is not intentional application logging. This keeps temporary debugging output out of the shipped code."

    elif rule == "leftover_debugger_statement":
        solution_type = "remove"
        replacement_code = current_code
        solution = "Remove this 'debugger' statement from the code. Otherwise execution can pause when developer tools are open."

    elif rule == "leftover_debug_print":
        solution_type = "remove"
        replacement_code = current_code
        solution = "Remove this print statement if it is only debugging output. Keep it only when the program intentionally uses it as user-facing or command-line output."

    elif rule == "unreachable_code":
        solution_type = "remove"
        replacement_code = current_code
        solution = "Remove this unreachable code. It cannot execute because the control flow always exits before reaching it."

    else:
        # Should not normally be reached - every known rule is handled above -
        # but keep a safe, honest fallback for any future/unknown rule.
        solution_type = "replace"
        solution = "Review the reported code and apply a manual fix based on the detected cause. An exact automatic replacement is not safe to generate from the available context."

    return {
        "error": finding.get("error", "Possible issue"),
        "bug_type": bug_type,
        "cause": cause or "Something in this code looks like it could cause a problem.",
        "why_occurs": "",
        "solution_type": solution_type,
        "solution": solution,
        "replacement_code": replacement_code,
        "add_location": add_location,
        "new_file_path": None,
        "explanation": "",
        "confidence": FALLBACK_CONFIDENCE_BY_RULE.get(rule, DEFAULT_FALLBACK_CONFIDENCE),
        "insufficient_evidence": True,
    }



def analyze_file(file_path: str, source: str) -> list:
    """
    Analyze a readable text file directly with Gemini when there is no
    specialized detector for its extension.

    This is intentionally separate from analyze_finding(): analyze_finding()
    receives one detector finding plus RAG context, while this function asks
    Gemini to inspect the whole file and return candidate findings that can
    then enter the normal RAG + report pipeline in main.py.

    Returns an empty list when no API key is available, the file is empty, the
    model call fails, or Gemini returns invalid JSON. This keeps the normal
    detector/fallback pipeline safe instead of inventing a finding.
    """
    if not source or not source.strip():
        return []

    api_keys = []

    # Reuse the same multi-key configuration as analyze_finding().
    for i in range(1, 11):
        key = os.getenv(f"GEMINI_API_KEY_{i}")
        if key and key.strip():
            api_keys.append(key.strip())

    old_key = os.getenv("GEMINI_API_KEY")
    if old_key and old_key.strip() and old_key.strip() not in api_keys:
        api_keys.append(old_key.strip())

    if not api_keys:
        return []

    api_keys = api_keys[:MAX_KEYS_TO_TRY_PER_CALL]

    file_prompt = f"""You are reviewing a source/configuration/text file for real software bugs.

File: {file_path}

Analyze the complete file below. Report ONLY issues that are reasonably supported
by the code/text itself. Do not invent bugs just because a style preference is
not followed. If there are no clear issues, return an empty JSON array.

For every real or strongly supported issue, return an object with exactly these
fields:
- error: short description of the bug
- bug_type: one of Runtime Error, Logic Error, Syntax Error, Type Error,
  Dependency Error, Security Issue, Performance Issue, API Error,
  Unnecessary Code, Other
- cause: why the issue occurs
- line_start: 1-based starting line number
- line_end: 1-based ending line number
- current_code: the smallest relevant code/text snippet
- function: function/class/component name if applicable, otherwise null

Rules:
- Use 1-based line numbers.
- Only report issues you can point to in the supplied file.
- Do not report vague possibilities without evidence.
- Do not include markdown fences or explanations outside the JSON array.

Return ONLY a JSON array.

FILE CONTENT:
--------------------
{source}
--------------------
"""

    payload = {
        "system_instruction": {
            "parts": [{"text": "You are a precise software bug detector. Return valid JSON only."}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": file_prompt}],
            }
        ],
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": 0.1,
            "maxOutputTokens": 2500,
            "topP": 0.9,
            "topK": 40,
        },
    }

    last_error = None

    for key_number, api_key in enumerate(api_keys, start=1):
        try:
            resp = requests.post(
                GEMINI_URL,
                params={"key": api_key},
                json=payload,
                timeout=GEMINI_REQUEST_TIMEOUT_SECONDS,
            )

            if resp.status_code != 200:
                last_error = (
                    f"Gemini file-analysis key {key_number} returned "
                    f"HTTP {resp.status_code}: {resp.text[:200]}"
                )
                continue

            data = resp.json()
            raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
            cleaned = (
                raw_text.strip()
                .removeprefix("```json")
                .removeprefix("```")
                .removesuffix("```")
                .strip()
            )

            result = json.loads(cleaned)
            if not isinstance(result, list):
                print("[llm_client] Gemini file analysis did not return a JSON array.")
                return []

            findings = []
            for item in result:
                if not isinstance(item, dict):
                    continue

                error = str(item.get("error") or "").strip()
                if not error:
                    continue

                bug_type = str(item.get("bug_type") or "Other").strip()
                allowed_types = {
                    "Runtime Error", "Logic Error", "Syntax Error", "Type Error",
                    "Dependency Error", "Security Issue", "Performance Issue",
                    "API Error", "Unnecessary Code", "Other",
                }
                if bug_type not in allowed_types:
                    bug_type = "Other"

                line_start = item.get("line_start")
                line_end = item.get("line_end")
                try:
                    line_start = int(line_start) if line_start is not None else None
                    line_end = int(line_end) if line_end is not None else line_start
                except (TypeError, ValueError):
                    line_start = None
                    line_end = None

                findings.append({
                    "error": error,
                    "bug_type": bug_type,
                    "cause": str(item.get("cause") or "").strip(),
                    "line_start": line_start,
                    "line_end": line_end,
                    "current_code": str(item.get("current_code") or "").strip(),
                    "function": item.get("function"),
                    "rule": "llm_file_analysis",
                    "file": file_path,
                })

            return findings

        except Exception as e:
            last_error = f"Gemini file analysis key {key_number} failed: {e}"

    if last_error:
        print(f"[llm_client] File analysis failed: {last_error}")
    return []

def get_fallback_report(finding: dict) -> dict:
    """
    Public entry point to the same real, rule-specific advice used when
    the LLM is unavailable - lets a caller intentionally skip the API call
    (e.g. once a per-scan call budget is used up) while still returning a
    genuine, specific answer instead of dropping the finding entirely.
    """
    return _fallback_report(finding)



def _normalize_solution_text(solution: str) -> str:
    """Keep the user-facing solution short: exactly two concise sentences when possible."""
    if not solution:
        return ""
    clean = re.sub(r"```(?:\w+)?|```", "", str(solution)).strip()
    clean = re.sub(r"\s+", " ", clean)
    sentences = re.split(r"(?<=[.!?])\s+", clean)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) >= 2:
        return " ".join(sentences[:2])
    return clean


def _validate_llm_result(result: dict, finding: dict) -> dict | None:
    """Validate and normalize the LLM result before it reaches the frontend."""
    if not isinstance(result, dict):
        return None

    solution_type = result.get("solution_type")
    replacement = result.get("replacement_code")

    if solution_type == "replace":
        if not isinstance(replacement, str) or not replacement.strip():
            return None
    elif solution_type in {"add", "remove", "create_file"}:
        if replacement is not None and not isinstance(replacement, str):
            result["replacement_code"] = str(replacement)

    solution = _normalize_solution_text(result.get("solution", ""))
    if not solution:
        return None

    result["solution"] = solution

    confidence = result.get("confidence")
    if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 100):
        result["confidence"] = 70
    else:
        result["confidence"] = int(confidence)

    result.setdefault("insufficient_evidence", False)
    return result


def analyze_finding(finding: dict, retrieved: list) -> dict:
    api_keys = []

    # Read multiple Gemini API keys from environment variables.
    for i in range(1, 11):
        key = os.getenv(f"GEMINI_API_KEY_{i}")
        if key and key.strip():
            api_keys.append(key.strip())

    # Keep support for the old single-key variable.
    old_key = os.getenv("GEMINI_API_KEY")
    if old_key and old_key.strip() and old_key.strip() not in api_keys:
        api_keys.append(old_key.strip())

    if not api_keys:
        return _fallback_report(finding)

    # Bounds the worst case for one finding to MAX_KEYS_TO_TRY_PER_CALL * the
    # per-attempt timeout below, no matter how many keys end up configured -
    # a scan with many findings calls this once per finding, so an unbounded
    # key list here directly multiplies into minutes of wall-clock time on a
    # bad run (several keys rate-limited or slow to respond) even before
    # accounting for how many findings there are.
    api_keys = api_keys[:MAX_KEYS_TO_TRY_PER_CALL]

    payload = {
        "system_instruction": {
            "parts": [{"text": SYSTEM_PROMPT}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": _build_user_message(
                            finding,
                            retrieved
                        )
                    }
                ],
            }
        ],
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": 0.1,  # Lower for more consistent/reliable outputs
            "maxOutputTokens": 1500,  # Increased for better solutions
            "topP": 0.9,
            "topK": 40,
        },
    }

    last_error = None
    last_429_response = None

    # Try each Gemini API key until one succeeds.
    for key_number, api_key in enumerate(api_keys, start=1):
        try:
            resp = requests.post(
                GEMINI_URL,
                params={"key": api_key},
                json=payload,
                timeout=GEMINI_REQUEST_TIMEOUT_SECONDS,
            )

            if resp.status_code == 200:
                data = resp.json()
                raw_text = data["candidates"][0]["content"]["parts"][0]["text"]

                try:
                    cleaned = (
                        raw_text
                        .strip()
                        .removeprefix("```json")
                        .removeprefix("```")
                        .removesuffix("```")
                        .strip()
                    )

                    result = json.loads(cleaned)

                    validated = _validate_llm_result(result, finding)
                    if validated is None:
                        print("[llm_client] Gemini returned an unusable fix. Using fallback.")
                        return _fallback_report(finding)

                    return validated

                except (json.JSONDecodeError, ValueError) as e:
                    print(f"[llm_client] Gemini returned non-JSON response: {raw_text[:300]}")
                    print(f"[llm_client] JSON parse error: {e}")
                    return _fallback_report(finding)

            if resp.status_code == 429:
                last_429_response = resp

            # Key failed - try the next key. Log the technical detail server-side
            # only - the person using the app should never see raw HTTP/API errors.
            last_error = f"Gemini key {key_number} returned HTTP {resp.status_code}: {resp.text[:200]}"

        except Exception as e:
            last_error = f"Gemini key {key_number} failed: {e}"

    # All keys failed. Print the real reason to the server logs (visible in Render's
    # Logs tab) so it can still be debugged, but keep the user-facing result simple.
    print(f"[llm_client] All Gemini API keys failed. Last error: {last_error}")

    fallback = _fallback_report(finding)

    # If every key failed specifically because of a rate limit, tell the person
    # clearly (and when they can try again) instead of the generic "fix it
    # yourself" message - this is shown once at the top of the scan results.
    if last_429_response is not None:
        retry_seconds, is_daily = _parse_rate_limit_info(last_429_response)
        fallback["rate_limited"] = True
        fallback["rate_limit_message"] = _build_rate_limit_message(retry_seconds, is_daily)

    return fallback
