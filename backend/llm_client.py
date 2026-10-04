"""
LLM client for Git Bug Detection.

Responsibilities
----------------
1. Analyze detector findings with Gemini.
2. Analyze complete source files when requested.
3. Use multiple Gemini API keys.
4. Rotate between configured keys.
5. Return structured JSON.
6. Provide safe fallback reports when Gemini is unavailable.

Environment variables
---------------------
GEMINI_API_KEY_1
GEMINI_API_KEY_2
GEMINI_API_KEY_3
GEMINI_API_KEY_4
GEMINI_API_KEY_5

The older GEMINI_API_KEY variable is also supported.
"""

import json
import math
import os
import re
import threading
import time

import requests


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.0-flash",
)

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/"
    f"v1beta/models/{GEMINI_MODEL}:generateContent"
)


# Five API keys are supported.
MAX_CONFIGURED_KEYS = 5

# Maximum time for one HTTP request.
GEMINI_REQUEST_TIMEOUT_SECONDS = 15

# Maximum output size for finding analysis.
MAX_OUTPUT_TOKENS_FINDING = 1800

# Maximum output size for direct file analysis.
MAX_OUTPUT_TOKENS_FILE = 3000


# ============================================================
# KEY ROTATION
# ============================================================

_KEY_LOCK = threading.Lock()

_NEXT_KEY_INDEX = 0


def _get_api_keys():
    """
    Load configured Gemini API keys.

    Preferred:
        GEMINI_API_KEY_1
        GEMINI_API_KEY_2
        ...
        GEMINI_API_KEY_5

    Older:
        GEMINI_API_KEY
    """

    keys = []

    for index in range(
        1,
        MAX_CONFIGURED_KEYS + 1,
    ):

        value = os.getenv(
            f"GEMINI_API_KEY_{index}"
        )

        if value and value.strip():

            key = value.strip()

            if key not in keys:
                keys.append(key)

    # Backward compatibility.
    old_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if (
        old_key
        and old_key.strip()
        and old_key.strip() not in keys
    ):

        keys.append(
            old_key.strip()
        )

    return keys


def _ordered_api_keys():
    """
    Return configured keys in rotating order.

    Example:

        first call:
        key1, key2, key3, key4, key5

        second call:
        key2, key3, key4, key5, key1

    This distributes requests instead of always starting
    with the first key.
    """

    global _NEXT_KEY_INDEX

    keys = _get_api_keys()

    if not keys:
        return []

    with _KEY_LOCK:

        start = (
            _NEXT_KEY_INDEX
            % len(keys)
        )

        _NEXT_KEY_INDEX = (
            _NEXT_KEY_INDEX + 1
        ) % len(keys)

    return (
        keys[start:]
        + keys[:start]
    )


# ============================================================
# RATE-LIMIT HELPERS
# ============================================================

def _parse_rate_limit_info(
    response,
):
    """
    Extract retry information from a Gemini
    429 response when available.
    """

    try:

        data = response.json()

    except (
        ValueError,
        json.JSONDecodeError,
    ):

        return None, False

    error = data.get(
        "error",
        {},
    )

    details = error.get(
        "details",
        [],
    )

    retry_seconds = None
    is_daily = False

    for detail in details:

        type_string = detail.get(
            "@type",
            "",
        )

        if type_string.endswith(
            "RetryInfo"
        ):

            delay = detail.get(
                "retryDelay",
                "",
            )

            try:

                retry_seconds = float(
                    str(delay).rstrip("s")
                )

            except (
                TypeError,
                ValueError,
            ):

                pass

        if type_string.endswith(
            "QuotaFailure"
        ):

            for violation in detail.get(
                "violations",
                [],
            ):

                combined = (
                    f"{violation.get('quotaId', '')} "
                    f"{violation.get('quotaMetric', '')}"
                )

                if (
                    "PerDay" in combined
                    or "per_day" in combined
                ):

                    is_daily = True

    return (
        retry_seconds,
        is_daily,
    )


def _build_rate_limit_message(
    retry_seconds,
    is_daily,
):
    if is_daily:

        if retry_seconds:

            minutes = max(
                1,
                math.ceil(
                    retry_seconds / 60
                ),
            )

            return (
                "The Gemini daily usage limit "
                "has been reached. Try again "
                f"in about {minutes} minute(s)."
            )

        return (
            "The Gemini daily usage limit "
            "has been reached. Please try "
            "again after the quota resets."
        )

    if retry_seconds:

        if retry_seconds < 60:

            seconds = (
                int(retry_seconds)
                + 3
            )

            return (
                "Gemini is temporarily "
                f"rate-limited. Try again "
                f"in about {seconds} seconds."
            )

        minutes = max(
            1,
            math.ceil(
                retry_seconds / 60
            ),
        )

        return (
            "Gemini is temporarily "
            f"rate-limited. Try again "
            f"in about {minutes} minute(s)."
        )

    return (
        "Gemini is temporarily "
        "rate-limited. Please try again."
    )


# ============================================================
# JSON CLEANING
# ============================================================

def _clean_json_text(
    raw_text,
):
    """
    Remove common markdown wrappers around JSON.
    """

    if not raw_text:
        return ""

    cleaned = raw_text.strip()

    if cleaned.startswith(
        "```json"
    ):

        cleaned = cleaned[
            len("```json"):
        ].strip()

    elif cleaned.startswith(
        "```"
    ):

        cleaned = cleaned[
            len("```"):
        ].strip()

    if cleaned.endswith(
        "```"
    ):

        cleaned = cleaned[
            :-3
        ].strip()

    return cleaned


def _parse_json(
    raw_text,
):
    cleaned = _clean_json_text(
        raw_text
    )

    try:

        return json.loads(
            cleaned
        )

    except (
        ValueError,
        json.JSONDecodeError,
    ):

        return None


# ============================================================
# GEMINI REQUEST
# ============================================================

def _request_gemini(
    payload,
):
    """
    Try the configured Gemini keys until one succeeds.

    Returns:

        {
            "ok": True,
            "data": ...,
            "key_number": ...
        }

    or:

        {
            "ok": False,
            "rate_limited": bool,
            "message": ...
        }
    """

    keys = _ordered_api_keys()

    if not keys:

        return {
            "ok": False,
            "rate_limited": False,
            "message": (
                "No Gemini API keys configured."
            ),
        }

    last_error = None
    last_429 = None

    for position, api_key in enumerate(
        keys,
        start=1,
    ):

        try:

            response = requests.post(
                GEMINI_URL,
                params={
                    "key": api_key,
                },
                json=payload,
                timeout=(
                    GEMINI_REQUEST_TIMEOUT_SECONDS
                ),
            )

            if response.status_code == 200:

                try:

                    return {
                        "ok": True,
                        "data": response.json(),
                        "key_number": position,
                    }

                except (
                    ValueError,
                    json.JSONDecodeError,
                ):

                    last_error = (
                        "Gemini returned invalid JSON."
                    )

                    continue

            if response.status_code == 429:

                last_429 = response

                last_error = (
                    f"Key {position} "
                    "was rate-limited."
                )

                # Move immediately to the next key.
                continue

            if response.status_code in (
                500,
                502,
                503,
                504,
            ):

                last_error = (
                    f"Key {position} "
                    f"returned HTTP "
                    f"{response.status_code}."
                )

                continue

            last_error = (
                f"Key {position} "
                f"returned HTTP "
                f"{response.status_code}: "
                f"{response.text[:200]}"
            )

        except requests.Timeout:

            last_error = (
                f"Key {position} "
                "timed out."
            )

            continue

        except requests.RequestException as exc:

            last_error = (
                f"Key {position} "
                f"request failed: {exc}"
            )

            continue

        except Exception as exc:

            last_error = (
                f"Key {position} "
                f"unexpected error: {exc}"
            )

            continue

    if last_429 is not None:

        retry_seconds, is_daily = (
            _parse_rate_limit_info(
                last_429
            )
        )

        return {
            "ok": False,
            "rate_limited": True,
            "message": _build_rate_limit_message(
                retry_seconds,
                is_daily,
            ),
        }

    return {
        "ok": False,
        "rate_limited": False,
        "message": (
            last_error
            or "All Gemini API keys failed."
        ),
    }


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an expert software debugging assistant.

Your job is to analyze the supplied code and determine whether
the reported issue is a genuine software problem.

Do not invent bugs.

For every genuine issue, provide:

1. cause
   Explain exactly why the code is problematic.

2. solution_type
   One of:
   - replace
   - add
   - remove
   - create_file

3. replacement_code
   If solution_type is replace, provide the complete corrected
   code that can directly replace the supplied problematic code.

4. solution
   Give a concrete, specific explanation of exactly what the
   developer should change.

5. explanation
   Explain why the proposed correction works.

6. confidence
   Integer from 0 to 100.
   This must represent confidence in BOTH:
   - the diagnosis
   - the proposed correction

7. insufficient_evidence
   true if the supplied context is not sufficient to make a
   reliable conclusion.

Important rules:

- Do not modify unrelated code.
- Do not invent missing variables or APIs.
- Preserve existing variable names where possible.
- A replacement solution MUST contain replacement_code.
- Do not give a generic solution when the supplied code allows
  a specific solution.
- Historical examples are evidence, not proof that the current
  bug is identical.
- If historical evidence conflicts with the supplied code,
  prioritize the supplied code.
- Return JSON only.
"""


# ============================================================
# FINDING PROMPT
# ============================================================

def _build_user_message(
    finding,
    retrieved,
):
    historical_parts = []

    for item in (
        retrieved or []
    ):

        record = item.get(
            "record",
            {},
        )

        historical_parts.append(
            "\n".join(
                [
                    f"Dataset: {record.get('dataset_source', 'Unknown')}",
                    f"Bug type: {record.get('bug_type', 'Unknown')}",
                    f"Description: {record.get('bug_description', '')}",
                    f"Historical solution: {record.get('solution', '')}",
                    (
                        "Similarity: "
                        f"{round(item.get('similarity', 0) * 100)}%"
                    ),
                ]
            )
        )

    historical_context = (
        "\n\n".join(
            historical_parts
        )
        if historical_parts
        else
        "(No similar historical bugs were retrieved.)"
    )

    return f"""
Analyze the following detected issue.

FILE:
{finding.get('file')}

FUNCTION:
{finding.get('function') or 'N/A'}

LINES:
{finding.get('line_start')} - {finding.get('line_end')}

BUG TYPE:
{finding.get('bug_type')}

DETECTOR:
{finding.get('rule')}

DETECTOR MESSAGE:
{finding.get('error')}

DETECTOR CAUSE:
{finding.get('cause')}

CURRENT CODE:
--------------------
{finding.get('current_code', '')}
--------------------

HISTORICAL RAG CONTEXT:
=======================
{historical_context}
=======================

Determine whether the detected problem is genuine.

If it is genuine:
- explain the exact cause
- provide a concrete solution
- provide corrected code when appropriate
- use historical context when it is relevant
- do not blindly copy a historical solution

If the detector appears to be wrong or there is insufficient
context, lower confidence and set insufficient_evidence to true.

Return ONLY the required JSON object.
"""


# ============================================================
# FALLBACK CONFIDENCE
# ============================================================

FALLBACK_CONFIDENCE_BY_RULE = {
    "eq_none": 90,
    "loose_equality": 90,
    "leftover_console_statement": 90,
    "leftover_debugger_statement": 90,
    "unreachable_code": 95,
    "bare_except": 85,
    "var_declaration": 85,
    "leftover_debug_print": 85,
    "unterminated_string": 75,
    "unterminated_template_literal": 75,
    "unterminated_comment": 75,
    "syntax_error": 70,
    "unclosed_bracket": 65,
    "mismatched_bracket": 65,
    "unexpected_closing_bracket": 65,
    "possibly_unused_function": 60,
    "mutable_default_arg": 55,
    "possible_division_by_zero": 50,
    "empty_catch_block": 55,
}

DEFAULT_FALLBACK_CONFIDENCE = 40


# ============================================================
# FALLBACK HELPERS
# ============================================================

BRACKET_CLOSER = {
    "(": ")",
    "[": "]",
    "{": "}",
}

UNCLOSED_RE = re.compile(
    r"Unclosed '(.+?)'"
)

MISMATCHED_RE = re.compile(
    r"found '(.+?)', expected '(.+?)'"
)

UNEXPECTED_RE = re.compile(
    r"Unexpected '(.+?)'"
)


def _fix_mutable_default(
    code,
):
    """
    Try a simple deterministic fix for:
        def f(x=[]):
    """

    if not code:
        return None

    match = re.search(
        r"(\w+)\s*=\s*(\[\]|\{\}|set\(\))",
        code,
    )

    if not match:
        return None

    name = match.group(1)
    original = match.group(2)

    replacement = (
        "[]" if original == "[]"
        else "{}" if original == "{}"
        else "set()"
    )

    updated = (
        code[:match.start()]
        + f"{name}=None"
        + code[match.end():]
    )

    lines = updated.splitlines()

    if not lines:
        return None

    indentation = "    "

    if len(lines) > 1:

        stripped = (
            lines[1].lstrip()
        )

        indentation = (
            lines[1][
                :len(lines[1])
                - len(stripped)
            ]
            or "    "
        )

    guard = [
        f"{indentation}if {name} is None:",
        f"{indentation}    {name} = {replacement}",
    ]

    return "\n".join(
        [lines[0]]
        + guard
        + lines[1:]
    )


# ============================================================
# FALLBACK REPORT
# ============================================================

def _fallback_report(
    finding,
):
    """
    Deterministic fallback used when Gemini cannot be reached.

    This is NOT presented as LLM reasoning.
    """

    rule = finding.get(
        "rule",
        "",
    )

    code = finding.get(
        "current_code",
        "",
    ) or ""

    error = finding.get(
        "error",
        "",
    ) or ""

    cause = finding.get(
        "cause",
        "",
    ) or ""

    solution_type = "add"
    solution = ""
    replacement_code = None
    add_location = None

    # --------------------------------------------------------
    # UNUSED FUNCTION
    # --------------------------------------------------------

    if rule == "possibly_unused_function":

        solution_type = "remove"

        solution = (
            "Remove this function if it is not intentionally "
            "used from another module. The detector could not "
            "find a call to it in the analyzed file."
        )

    # --------------------------------------------------------
    # BARE EXCEPT
    # --------------------------------------------------------

    elif rule == "bare_except":

        solution_type = "replace"

        replacement_code = code.replace(
            "except:",
            "except Exception as e:",
            1,
        )

        solution = (
            "Replace the bare except with "
            "'except Exception as e:'. A bare except also "
            "catches system-level exceptions such as "
            "KeyboardInterrupt and SystemExit."
        )

    # --------------------------------------------------------
    # MUTABLE DEFAULT
    # --------------------------------------------------------

    elif rule == "mutable_default_arg":

        replacement_code = (
            _fix_mutable_default(
                code
            )
        )

        if replacement_code:

            solution_type = "replace"

            solution = (
                "Use None as the default and create a new "
                "list, dictionary, or set inside the function. "
                "The current mutable default object can be "
                "shared between multiple function calls."
            )

        else:

            solution_type = "add"

            solution = (
                "Change the mutable default to None and "
                "initialize a new list, dictionary, or set "
                "inside the function before using it."
            )

    # --------------------------------------------------------
    # NONE COMPARISON
    # --------------------------------------------------------

    elif rule == "eq_none":

        solution_type = "replace"

        replacement_code = (
            code
            .replace(
                "== None",
                "is None",
            )
            .replace(
                "!= None",
                "is not None",
            )
        )

        solution = (
            "Use 'is None' or 'is not None' instead of "
            "equality operators when checking against None."
        )

    # --------------------------------------------------------
    # DIVISION BY ZERO
    # --------------------------------------------------------

    elif rule == "possible_division_by_zero":

        solution_type = "add"

        add_location = (
            "Add the validation immediately before "
            "the division."
        )

        solution = (
            "Check that the denominator is not zero before "
            "performing the division. Decide whether the "
            "program should return a default value, skip the "
            "operation, or raise a meaningful error."
        )

    # --------------------------------------------------------
    # SYNTAX ERROR
    # --------------------------------------------------------

    elif rule == "syntax_error":

        solution_type = "replace"

        solution = (
            "Correct the syntax problem reported by the "
            "parser. The exact parser message is: "
            f"{cause}"
        )

    # --------------------------------------------------------
    # UNCLOSED BRACKET
    # --------------------------------------------------------

    elif rule == "unclosed_bracket":

        solution_type = "replace"

        match = UNCLOSED_RE.search(
            error
        )

        opener = (
            match.group(1)
            if match
            else "{"
        )

        closer = BRACKET_CLOSER.get(
            opener,
            "}",
        )

        solution = (
            f"Add the missing '{closer}' corresponding "
            f"to the opening '{opener}'. Check the surrounding "
            "block or expression to ensure the brackets are "
            "properly balanced."
        )

    # --------------------------------------------------------
    # MISMATCHED BRACKET
    # --------------------------------------------------------

    elif rule == "mismatched_bracket":

        solution_type = "replace"

        match = MISMATCHED_RE.search(
            error
        )

        if match:

            found = match.group(1)
            expected = match.group(2)

            solution = (
                f"The code contains '{found}' where "
                f"'{expected}' is expected. Correct the "
                "closing bracket or check an earlier opening "
                "bracket that may have been closed incorrectly."
            )

        else:

            solution = (
                "Correct the mismatched opening and closing "
                "brackets around this code."
            )

    # --------------------------------------------------------
    # UNEXPECTED BRACKET
    # --------------------------------------------------------

    elif rule == "unexpected_closing_bracket":

        solution_type = "remove"

        match = UNEXPECTED_RE.search(
            error
        )

        character = (
            match.group(1)
            if match
            else "bracket"
        )

        solution = (
            f"Remove the unexpected '{character}' if it "
            "does not belong to the surrounding expression. "
            "Otherwise add the corresponding opening bracket "
            "at the correct location."
        )

    # --------------------------------------------------------
    # UNTERMINATED STRING
    # --------------------------------------------------------

    elif rule == "unterminated_string":

        solution_type = "replace"

        solution = (
            "Add the missing closing quote so that the string "
            "is properly terminated."
        )

    # --------------------------------------------------------
    # TEMPLATE LITERAL
    # --------------------------------------------------------

    elif rule == "unterminated_template_literal":

        solution_type = "replace"

        solution = (
            "Add the missing closing backtick to terminate "
            "the template literal."
        )

    # --------------------------------------------------------
    # COMMENT
    # --------------------------------------------------------

    elif rule == "unterminated_comment":

        solution_type = "replace"

        solution = (
            "Add the missing closing */ so that the block "
            "comment terminates correctly."
        )

    # --------------------------------------------------------
    # JAVASCRIPT LOOSE EQUALITY
    # --------------------------------------------------------

    elif rule == "loose_equality":

        solution_type = "replace"

        if "!=" in code:

            replacement_code = code.replace(
                "!=",
                "!==",
                1,
            )

            solution = (
                "Replace != with !== so the comparison "
                "checks both type and value."
            )

        else:

            replacement_code = code.replace(
                "==",
                "===",
                1,
            )

            solution = (
                "Replace == with === so the comparison "
                "checks both type and value."
            )

    # --------------------------------------------------------
    # VAR
    # --------------------------------------------------------

    elif rule == "var_declaration":

        solution_type = "replace"

        replacement_code = code.replace(
            "var ",
            "let ",
            1,
        )

        solution = (
            "Replace var with let, or const if the value "
            "is never reassigned, to give the variable "
            "block scope."
        )

    # --------------------------------------------------------
    # EMPTY CATCH
    # --------------------------------------------------------

    elif rule == "empty_catch_block":

        solution_type = "add"

        solution = (
            "Add appropriate error handling or logging "
            "inside this catch block. Silently ignoring "
            "an exception makes failures difficult to "
            "diagnose."
        )

    # --------------------------------------------------------
    # DEBUGGING STATEMENT
    # --------------------------------------------------------

    elif rule == "leftover_console_statement":

        solution_type = "remove"

        solution = (
            "Remove this debugging console statement unless "
            "it is intentionally required as application "
            "output."
        )

    elif rule == "leftover_debugger_statement":

        solution_type = "remove"

        solution = (
            "Remove the debugger statement before deploying "
            "the application."
        )

    elif rule == "leftover_debug_print":

        solution_type = "remove"

        solution = (
            "Remove the debugging print statement unless "
            "it is intentionally part of the application's "
            "normal output."
        )

    # --------------------------------------------------------
    # UNREACHABLE CODE
    # --------------------------------------------------------

    elif rule == "unreachable_code":

        solution_type = "remove"

        solution = (
            "Remove the unreachable code because execution "
            "cannot reach this section under the detected "
            "control flow."
        )

    # --------------------------------------------------------
    # UNKNOWN
    # --------------------------------------------------------

    else:

        solution_type = "add"

        solution = (
            "The detector identified a possible issue, but "
            "there is not enough information to generate a "
            "safe automatic replacement. Review the reported "
            "cause and verify the surrounding code."
        )

    confidence = (
        FALLBACK_CONFIDENCE_BY_RULE.get(
            rule,
            DEFAULT_FALLBACK_CONFIDENCE,
        )
    )

    return {
        "error": (
            error
            or "Possible issue"
        ),

        "bug_type": finding.get(
            "bug_type",
            "Other",
        ),

        "cause": (
            cause
            or "The detector identified a possible "
               "problem in this code."
        ),

        "why_occurs": "",

        "solution_type": solution_type,

        "solution": solution,

        "replacement_code": replacement_code,

        "add_location": add_location,

        "new_file_path": None,

        "explanation": "",

        "confidence": confidence,

        "insufficient_evidence": True,
    }


# ============================================================
# ANALYZE DETECTOR FINDING
# ============================================================

def analyze_finding(
    finding: dict,
    retrieved: list,
):
    """
    Analyze one detector finding with Gemini.

    RAG results are included as historical context.
    """

    api_keys = _get_api_keys()

    if not api_keys:

        return _fallback_report(
            finding
        )

    payload = {
        "system_instruction": {
            "parts": [
                {
                    "text": SYSTEM_PROMPT
                }
            ]
        },

        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": _build_user_message(
                            finding,
                            retrieved,
                        )
                    }
                ],
            }
        ],

        "generationConfig": {
            "response_mime_type": (
                "application/json"
            ),

            "temperature": 0.1,

            "maxOutputTokens": (
                MAX_OUTPUT_TOKENS_FINDING
            ),

            "topP": 0.9,

            "topK": 40,
        },
    }

    response = _request_gemini(
        payload
    )

    if not response.get(
        "ok"
    ):

        fallback = _fallback_report(
            finding
        )

        if response.get(
            "rate_limited"
        ):

            fallback[
                "rate_limited"
            ] = True

            fallback[
                "rate_limit_message"
            ] = response.get(
                "message"
            )

        print(
            "[llm_client] "
            f"Gemini finding analysis failed: "
            f"{response.get('message')}"
        )

        return fallback

    try:

        data = response[
            "data"
        ]

        raw_text = (
            data[
                "candidates"
            ][0][
                "content"
            ][
                "parts"
            ][0][
                "text"
            ]
        )

    except (
        KeyError,
        IndexError,
        TypeError,
    ):

        print(
            "[llm_client] Gemini response "
            "did not contain expected content."
        )

        return _fallback_report(
            finding
        )

    result = _parse_json(
        raw_text
    )

    if not isinstance(
        result,
        dict,
    ):

        print(
            "[llm_client] Gemini returned "
            "invalid finding JSON."
        )

        return _fallback_report(
            finding
        )

    # --------------------------------------------------------
    # NORMALIZE RESULT
    # --------------------------------------------------------

    allowed_bug_types = {
        "Runtime Error",
        "Logic Error",
        "Syntax Error",
        "Type Error",
        "Dependency Error",
        "Security Issue",
        "Performance Issue",
        "API Error",
        "Unnecessary Code",
        "Other",
    }

    bug_type = str(
        result.get(
            "bug_type",
            finding.get(
                "bug_type",
                "Other",
            ),
        )
    )

    if bug_type not in allowed_bug_types:

        bug_type = "Other"

    result["bug_type"] = bug_type

    solution_type = result.get(
        "solution_type",
        "add",
    )

    if solution_type not in {
        "replace",
        "add",
        "remove",
        "create_file",
    }:

        solution_type = "add"

    result[
        "solution_type"
    ] = solution_type

    # --------------------------------------------------------
    # VALIDATE SOLUTION
    # --------------------------------------------------------

    solution = str(
        result.get(
            "solution",
            "",
        )
        or ""
    ).strip()

    if not solution:

        print(
            "[llm_client] Gemini returned "
            "an empty solution."
        )

        return _fallback_report(
            finding
        )

    result[
        "solution"
    ] = solution

    if solution_type == "replace":

        replacement = str(
            result.get(
                "replacement_code",
                "",
            )
            or ""
        )

        if not replacement.strip():

            print(
                "[llm_client] Gemini returned "
                "a replace solution without "
                "replacement code."
            )

            return _fallback_report(
                finding
            )

        result[
            "replacement_code"
        ] = replacement

    else:

        if result.get(
            "replacement_code"
        ) is None:

            result[
                "replacement_code"
            ] = None

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    confidence = result.get(
        "confidence",
        70,
    )

    try:

        confidence = int(
            float(confidence)
        )

    except (
        TypeError,
        ValueError,
    ):

        confidence = 70

    confidence = max(
        0,
        min(
            100,
            confidence,
        ),
    )

    result[
        "confidence"
    ] = confidence

    result[
        "insufficient_evidence"
    ] = bool(
        result.get(
            "insufficient_evidence",
            False,
        )
    )

    return result


# ============================================================
# DIRECT FILE ANALYSIS
# ============================================================

def analyze_file(
    file_path: str,
    source: str,
):
    """
    Ask Gemini to inspect a complete readable source file.

    This is used for files without a specialized local detector.

    Very large files should eventually be chunked by the repository
    worker before calling this function. The function itself is
    intentionally focused on one analysis unit.
    """

    if not source or not source.strip():

        return []

    prompt = f"""
You are analyzing a source-code file for genuine software bugs.

FILE:
{file_path}

Analyze the supplied code carefully.

Find only bugs that are reasonably supported by the code itself.

Look for:
- syntax problems
- runtime errors
- logic errors
- incorrect API usage
- type problems
- security problems
- dependency problems
- performance problems
- incorrect error handling
- unreachable code
- clearly unnecessary code that can cause a problem

Do NOT report:
- simple style preferences
- formatting preferences
- vague possibilities
- issues that cannot be supported by the supplied code

For every genuine issue return:

{{
  "error": "short bug description",
  "bug_type": "Runtime Error",
  "cause": "exact reason",
  "line_start": 1,
  "line_end": 1,
  "current_code": "smallest relevant code",
  "function": "function/class name or null"
}}

Allowed bug_type values:

Runtime Error
Logic Error
Syntax Error
Type Error
Dependency Error
Security Issue
Performance Issue
API Error
Unnecessary Code
Other

Use 1-based line numbers.

Return ONLY a JSON array.

FILE CONTENT:
==================================================
{source}
==================================================
"""

    payload = {
        "system_instruction": {
            "parts": [
                {
                    "text": (
                        "You are a precise software "
                        "bug detector. Return JSON only."
                    )
                }
            ]
        },

        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": prompt
                    }
                ],
            }
        ],

        "generationConfig": {
            "response_mime_type": (
                "application/json"
            ),

            "temperature": 0.1,

            "maxOutputTokens": (
                MAX_OUTPUT_TOKENS_FILE
            ),

            "topP": 0.9,

            "topK": 40,
        },
    }

    response = _request_gemini(
        payload
    )

    if not response.get(
        "ok"
    ):

        print(
            "[llm_client] "
            f"Direct file analysis failed: "
            f"{response.get('message')}"
        )

        return []

    try:

        raw_text = (
            response[
                "data"
            ][
                "candidates"
            ][0][
                "content"
            ][
                "parts"
            ][0][
                "text"
            ]
        )

    except (
        KeyError,
        IndexError,
        TypeError,
    ):

        return []

    parsed = _parse_json(
        raw_text
    )

    if not isinstance(
        parsed,
        list,
    ):

        return []

    findings = []

    allowed_bug_types = {
        "Runtime Error",
        "Logic Error",
        "Syntax Error",
        "Type Error",
        "Dependency Error",
        "Security Issue",
        "Performance Issue",
        "API Error",
        "Unnecessary Code",
        "Other",
    }

    for item in parsed:

        if not isinstance(
            item,
            dict,
        ):
            continue

        error = str(
            item.get(
                "error",
                "",
            )
            or ""
        ).strip()

        if not error:
            continue

        bug_type = str(
            item.get(
                "bug_type",
                "Other",
            )
            or "Other"
        ).strip()

        if bug_type not in allowed_bug_types:

            bug_type = "Other"

        line_start = item.get(
            "line_start"
        )

        line_end = item.get(
            "line_end"
        )

        try:

            line_start = (
                int(line_start)
                if line_start is not None
                else None
            )

        except (
            TypeError,
            ValueError,
        ):

            line_start = None

        try:

            line_end = (
                int(line_end)
                if line_end is not None
                else line_start
            )

        except (
            TypeError,
            ValueError,
        ):

            line_end = line_start

        findings.append(
            {
                "error": error,

                "bug_type": bug_type,

                "cause": str(
                    item.get(
                        "cause",
                        "",
                    )
                    or ""
                ).strip(),

                "line_start": line_start,

                "line_end": line_end,

                "current_code": str(
                    item.get(
                        "current_code",
                        "",
                    )
                    or ""
                ).strip(),

                "function": item.get(
                    "function"
                ),

                "rule": (
                    "llm_file_analysis"
                ),

                "file": file_path,
            }
        )

    return findings


# ============================================================
# PUBLIC FALLBACK
# ============================================================

def get_fallback_report(
    finding: dict,
):
    """
    Public fallback function used by main.py.
    """

    return _fallback_report(
        finding
    )
