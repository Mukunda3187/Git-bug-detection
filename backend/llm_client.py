"""
LLM client for RAG GitHub Bug Detection and Recovery.

Used by main.py:
    analyze_finding()
    analyze_file()
    get_fallback_report()
"""

import json
import math
import os
import re
from typing import Any, Dict, List

import requests


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

GEMINI_MODEL = "gemini-3.8-flash"

GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)

MAX_KEYS_TO_TRY_PER_CALL = 2

GEMINI_REQUEST_TIMEOUT_SECONDS = 15


# ============================================================
# ALLOWED BUG TYPES
# ============================================================

ALLOWED_BUG_TYPES = {
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


# ============================================================
# API KEYS
# ============================================================

def _get_api_keys() -> List[str]:

    keys = []

    for i in range(1, 11):

        key = os.getenv(
            f"GEMINI_API_KEY_{i}"
        )

        if key and key.strip():

            key = key.strip()

            if key not in keys:
                keys.append(key)

    key = os.getenv(
        "GEMINI_API_KEY"
    )

    if key and key.strip():

        key = key.strip()

        if key not in keys:
            keys.append(key)

    return keys[:MAX_KEYS_TO_TRY_PER_CALL]


# ============================================================
# JSON CLEANING
# ============================================================

def _clean_json(text: str) -> str:

    text = (text or "").strip()

    if text.startswith("```json"):
        text = text[7:]

    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    return text.strip()


# ============================================================
# CLEAN REPLACEMENT CODE
# ============================================================

def _clean_replacement_code(code: str) -> str:
    """
    Clean LLM-generated replacement code.

    The frontend should receive ONLY source code.

    Removes:
        - Markdown code fences
        - Lines 10-20
        - Function: xyz
        - Function xyz
        - File: xyz
        - Solution:
        - Corrected Code:
        - Replace with this code
        - Explanation:
        - other obvious metadata lines

    It intentionally does NOT remove real source declarations such as:

        function getGenAI() {

        class User {

        def get_user():

    because those contain actual source syntax.
    """

    if not isinstance(code, str):
        return ""

    code = code.strip("\r\n")

    if not code:
        return ""

    # --------------------------------------------------------
    # Remove markdown fences
    # --------------------------------------------------------

    code = re.sub(
        r"^\s*```[A-Za-z0-9_+#.\-]*\s*\n?",
        "",
        code,
        flags=re.IGNORECASE
    )

    code = re.sub(
        r"\n?\s*```\s*$",
        "",
        code
    )

    code = code.strip("\r\n")

    if not code:
        return ""

    lines = code.splitlines()

    cleaned_lines = []

    for line in lines:

        stripped = line.strip()

        # Empty line
        if not stripped:
            cleaned_lines.append(line)
            continue

        # ----------------------------------------------------
        # Line range metadata
        # ----------------------------------------------------

        if re.match(
            r"^Lines?\s+\d+\s*[-–—]\s*\d+\s*$",
            stripped,
            re.IGNORECASE
        ):
            continue

        if re.match(
            r"^Lines?\s*:\s*\d+\s*[-–—]\s*\d+\s*$",
            stripped,
            re.IGNORECASE
        ):
            continue

        # ----------------------------------------------------
        # Metadata labels
        # ----------------------------------------------------

        if re.match(
            r"^(Function|Method|File|Filename|Path)\s*:\s*.+$",
            stripped,
            re.IGNORECASE
        ):
            continue

        if re.match(
            r"^(Solution|Corrected Code|Replacement Code|Replace With This Code)\s*:?\s*$",
            stripped,
            re.IGNORECASE
        ):
            continue

        if re.match(
            r"^(Explanation|Description|Reason|Cause)\s*:?\s*$",
            stripped,
            re.IGNORECASE
        ):
            continue

        # ----------------------------------------------------
        # Metadata such as:
        #
        # function getGenAI
        # class User
        # method calculate
        #
        # BUT DO NOT remove:
        #
        # function getGenAI() {
        # class User {
        # def calculate():
        # ----------------------------------------------------

        if re.match(
            r"^(function|method|class|file)\s+[A-Za-z_$][\w$.\-]*\s*$",
            stripped,
            re.IGNORECASE
        ):
            continue

        # ----------------------------------------------------
        # "REPLACE WITH THIS CODE"
        # ----------------------------------------------------

        if re.match(
            r"^(replace|replace with|use|corrected code|replacement code)\s+"
            r"(with\s+)?(this\s+)?code\s*:?\s*$",
            stripped,
            re.IGNORECASE
        ):
            continue

        # ----------------------------------------------------
        # Markdown headings
        # ----------------------------------------------------

        if re.match(
            r"^#{1,6}\s*(corrected code|solution|replacement|code)\s*$",
            stripped,
            re.IGNORECASE
        ):
            continue

        cleaned_lines.append(line)

    result = "\n".join(
        cleaned_lines
    ).strip("\r\n")

    # --------------------------------------------------------
    # Remove accidental outer quotation marks
    # --------------------------------------------------------

    if (
        len(result) >= 2
        and result[0] == '"'
        and result[-1] == '"'
    ):
        result = result[1:-1].strip()

    return result


# ============================================================
# RATE LIMIT INFORMATION
# ============================================================

def _parse_rate_limit_info(
    response
):

    try:
        data = response.json()

    except Exception:
        return None, False

    retry_seconds = None
    is_daily = False

    details = (
        data
        .get("error", {})
        .get("details", [])
    )

    for detail in details:

        type_name = str(
            detail.get(
                "@type",
                ""
            )
        )

        if type_name.endswith(
            "RetryInfo"
        ):

            delay = str(
                detail.get(
                    "retryDelay",
                    ""
                )
            ).rstrip("s")

            try:
                retry_seconds = float(
                    delay
                )

            except ValueError:
                pass

        if type_name.endswith(
            "QuotaFailure"
        ):

            for violation in detail.get(
                "violations",
                []
            ):

                text = (
                    f"{violation.get('quotaId', '')} "
                    f"{violation.get('quotaMetric', '')}"
                )

                if (
                    "PerDay" in text
                    or "per_day" in text
                ):
                    is_daily = True

    return retry_seconds, is_daily


def _build_rate_limit_message(
    retry_seconds,
    is_daily
):

    if is_daily:

        if retry_seconds:

            minutes = max(
                1,
                math.ceil(
                    retry_seconds / 60
                )
            )

            return (
                "Gemini daily quota reached. "
                f"Try again in about {minutes} minute(s)."
            )

        return (
            "Gemini daily quota reached. "
            "The local repair engine was used."
        )

    if retry_seconds:

        if retry_seconds < 60:

            return (
                "Gemini is temporarily rate-limited. "
                f"Try again in about {int(retry_seconds) + 5} seconds."
            )

        minutes = max(
            1,
            math.ceil(
                retry_seconds / 60
            )
        )

        return (
            "Gemini is temporarily rate-limited. "
            f"Try again in about {minutes} minute(s)."
        )

    return (
        "Gemini is temporarily unavailable. "
        "The local repair engine was used."
    )


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = r"""
You are an expert software bug detection and repair engine.

A bug detector has already identified a candidate bug.

Your job is to:

1. Understand the bug.
2. Explain the cause.
3. Produce corrected code.

CRITICAL RULE:

replacement_code MUST ALWAYS contain actual copy-pasteable source code.

replacement_code must contain ONLY source code.

DO NOT include inside replacement_code:

- function names as labels
- file names
- line numbers
- "Lines 54-63"
- "Function:"
- "File:"
- "Solution:"
- "Corrected Code:"
- "Replacement Code:"
- "REPLACE WITH THIS CODE"
- explanations
- descriptions
- markdown headings
- markdown ``` fences

The replacement_code field must start directly with the first line
of the corrected source code and end with the last line of the
corrected source code.

For syntax errors:
- fix indentation
- fix brackets
- fix quotes
- fix invalid syntax
- return the corrected surrounding code

For unnecessary code:
- return the surrounding code with the unnecessary line removed

For security/runtime/logic bugs:
- return the corrected relevant code block

For add/check fixes:
- return the original code with the required check inserted

Do not return instructions such as:

"add indentation"

"remove this line"

"change this variable"

Instead return the actual corrected code.
1. cause:
   - Carefully examine the actual code in this finding.
   - Explain what is wrong, where the problem happens, and why it happens.
   - Use very simple, everyday English that a 5-year-old could understand.
   - Write a natural paragraph. Use as many sentences as needed.
   - Mention the actual variable, condition, function, or operation from
     THIS code cell.
   - Explain what bad thing could happen when this code runs.
   - CRITICAL: Every cause must be written independently for this exact
     code cell. Never copy or reuse a cause from another bug, even if the
     bug type is the same. Two different bugs must never share the same
     cause text.
   - Avoid difficult technical words. If a technical word is necessary,
     explain what it means in simple English.
   - Never guess or invent details that are not supported by the code.

2. solution:
   - Explain how to fix the exact problem found in this code.
   - Use very simple English that a 5-year-old could understand.
   - Write a natural paragraph and use as many sentences as needed.
   - Explain what needs to change, where to change it, and why the change
     will solve the problem.
   - Give clear steps in words, not vague advice such as "fix the error"
     or "check the code".
   - CRITICAL: Every solution must be written independently for this exact
     code cell. Never reuse a generic solution just because another finding
     has the same bug type. Two different bugs must never share the same
     solution text.
   - If the code does not provide enough information for a safe fix,
     explain what is missing instead of guessing.
   - Do not include replacement code in this paragraph. Put code in
     replacement_code.
Return ONLY valid JSON:

{
    "error": "...",
    "bug_type": "...",
    "cause": "...",
    "why_occurs": "...",
    "solution_type": "replace",
    "solution": "...",
    "replacement_code": "...",
    "add_location": null,
    "new_file_path": null,
    "explanation": "...",
    "confidence": 0,
    "insufficient_evidence": false
}

Confidence:

90-100 = exact and obvious fix
75-89 = strong fix
60-74 = reasonable fix
below 60 = uncertain

Even with low confidence, provide the best concrete
replacement_code possible from the supplied code.
"""


# ============================================================
# CODE CONTEXT
# ============================================================

def _get_code_context(
    finding: Dict[str, Any]
) -> str:

    # code_chunker.py context has priority.
    for key in (
        "chunk_context",
        "code_chunk",
        "current_code"
    ):

        value = finding.get(
            key
        )

        if (
            isinstance(value, str)
            and value.strip()
        ):

            return value.strip()

    return ""


# ============================================================
# RAG PROMPT
# ============================================================

def _build_user_message(
    finding: Dict[str, Any],
    retrieved: List[dict]
) -> str:

    dataset_items = []
    github_items = []

    for item in retrieved or []:

        record = item.get(
            "record",
            {}
        ) or {}

        similarity = float(
            item.get(
                "similarity",
                0
            ) or 0
        )

        source = str(
            record.get(
                "dataset_source"
            )
            or record.get(
                "source"
            )
            or record.get(
                "source_type"
            )
            or ""
        ).lower()

        artifact_type = str(
            record.get(
                "artifact_type"
            )
            or record.get(
                "type"
            )
            or ""
        ).lower()

        is_github = (
            "github" in source
            or "issue" in source
            or "pull request" in source
            or "pull_request" in source
            or artifact_type in {
                "issue",
                "pull_request",
                "pr"
            }
            or "github" in artifact_type
        )

        if is_github:

            github_items.append(
                (
                    record,
                    similarity
                )
            )

        else:

            dataset_items.append(
                (
                    record,
                    similarity
                )
            )

    # --------------------------------------------------------
    # LOCAL DATASET
    # --------------------------------------------------------

    dataset_text = []

    for record, similarity in dataset_items:

        dataset_text.append(
            f"""
Source:
{record.get('dataset_source', 'local dataset')}

Bug Type:
{record.get('bug_type', 'Unknown')}

Description:
{record.get('bug_description', 'N/A')}

Previous Solution:
{record.get('solution', 'N/A')}

Similarity:
{similarity * 100:.1f}%
"""
        )

    if not dataset_text:

        dataset_text = [
            "(No similar local bugs found.)"
        ]

    # --------------------------------------------------------
    # GITHUB HISTORY
    # --------------------------------------------------------

    github_text = []

    for record, similarity in github_items:

        labels = record.get(
            "labels",
            []
        ) or []

        label_names = []

        if isinstance(
            labels,
            list
        ):

            for label in labels:

                if isinstance(
                    label,
                    dict
                ):

                    if label.get(
                        "name"
                    ):

                        label_names.append(
                            str(
                                label["name"]
                            )
                        )

                elif label:

                    label_names.append(
                        str(label)
                    )

        labels_text = (
            ", ".join(
                label_names
            )
            if label_names
            else "None"
        )

        github_text.append(
            f"""
Artifact Type:
{record.get('artifact_type') or record.get('type') or 'GitHub artifact'}

Title:
{record.get('title') or 'Untitled'}

State:
{record.get('state') or 'unknown'}

Labels:
{labels_text}

URL:
{record.get('url') or record.get('html_url') or 'N/A'}

Similarity:
{similarity * 100:.1f}%

Description:
{str(record.get('body') or '(none)')[:4000]}

Discussion:
{str(record.get('comments') or '(none)')[:3000]}

Previous Patch:
{str(record.get('patch') or '(none)')[:5000]}
"""
        )

    if not github_text:

        github_text = [
            "(No historical GitHub Issues or Pull Requests found.)"
        ]

    # --------------------------------------------------------
    # CURRENT CODE
    # --------------------------------------------------------

    code_context = _get_code_context(
        finding
    )

    current_code = str(
        finding.get(
            "current_code",
            ""
        ) or ""
    )

    return f"""
==================================================
CURRENT BUG
==================================================

File:
{finding.get('file')}

Function:
{finding.get('function') or 'N/A'}

Lines:
{finding.get('line_start')} - {finding.get('line_end')}

Rule:
{finding.get('rule')}

Bug Type:
{finding.get('bug_type')}

Detector Error:
{finding.get('error')}

Detector Cause:
{finding.get('cause')}


==================================================
FULL RELEVANT CODE CHUNK
==================================================

{code_context}


==================================================
REPORTED CODE
==================================================

{current_code}


==================================================
LOCAL RAG EVIDENCE
==================================================

{''.join(dataset_text)}


==================================================
HISTORICAL GITHUB EVIDENCE
==================================================

{''.join(github_text)}


==================================================
REPAIR TASK
==================================================

Analyze this exact bug.

Then provide a concrete corrected-code replacement.

IMPORTANT:

replacement_code MUST contain ONLY actual source code.

Do NOT include:

Function name labels
File names
Line numbers
"Lines 54-63"
"Function:"
"Solution:"
"Corrected Code:"
"REPLACE WITH THIS CODE"
Explanations
Markdown fences

For an indentation error, return the corrected
code with the proper indentation.

For a removal, return the code after removing
the unnecessary line.

For an insertion/check, return the code with
the check inserted.

Do not return only an explanation.

Use historical evidence as supporting evidence.
Do not blindly copy historical patches.
"""


# ============================================================
# PYTHON SYNTAX REPAIR
# ============================================================

def _fix_python_syntax_error(
    finding
):

    code = _get_code_context(
        finding
    )

    if not code:
        return None

    error_text = str(
        finding.get(
            "error",
            ""
        )
    )

    cause = str(
        finding.get(
            "cause",
            ""
        )
    )

    combined = (
        error_text
        + " "
        + cause
    ).lower()

    lines = code.splitlines()

    # --------------------------------------------------------
    # INDENTATION
    # --------------------------------------------------------

    if (
        "indent" in combined
        or "expected an indented block"
        in combined
        or any(
            line.strip().endswith(":")
            for line in lines
        )
    ):

        if len(lines) == 1:

            stripped = (
                lines[0].lstrip()
            )

            if stripped:

                return (
                    "    "
                    + stripped
                )

        fixed_lines = list(
            lines
        )

        for i in range(
            1,
            len(fixed_lines)
        ):

            current = (
                fixed_lines[i]
            )

            previous = (
                fixed_lines[i - 1]
                .strip()
            )

            if (
                current.strip()
                and not current.startswith(
                    (" ", "\t")
                )
                and previous.endswith(":")
            ):

                prev_indent = len(
                    fixed_lines[i - 1]
                ) - len(
                    fixed_lines[i - 1].lstrip()
                )

                fixed_lines[i] = (
                    " " * (prev_indent + 4)
                    + current.lstrip()
                )

        fixed = "\n".join(
            fixed_lines
        )

        if fixed != code:
            return fixed

    # --------------------------------------------------------
    # UNCLOSED BRACKET
    # --------------------------------------------------------

    match = re.search(
        r"Unclosed '([(\[{])'",
        error_text
    )

    if match:

        closing = {
            "(": ")",
            "[": "]",
            "{": "}"
        }.get(
            match.group(1)
        )

        if closing:

            return (
                code.rstrip()
                + closing
            )

    # --------------------------------------------------------
    # UNEXPECTED CLOSING BRACKET
    # --------------------------------------------------------

    match = re.search(
        r"Unexpected '([)\]}])'",
        error_text
    )

    if match:

        character = (
            match.group(1)
        )

        position = code.rfind(
            character
        )

        if position >= 0:

            return (
                code[:position]
                + code[position + 1:]
            )

    # --------------------------------------------------------
    # MISMATCHED BRACKET
    # --------------------------------------------------------

    match = re.search(
        r"expected ['\"]?([)\]}])",
        error_text
    )

    if match:

        expected = (
            match.group(1)
        )

        for character in (
            ")",
            "]",
            "}"
        ):

            position = code.rfind(
                character
            )

            if position >= 0:

                return (
                    code[:position]
                    + expected
                    + code[position + 1:]
                )

    # --------------------------------------------------------
    # UNTERMINATED STRING
    # --------------------------------------------------------

    if (
        "unterminated string"
        in combined
        or "eol while scanning string"
        in combined
    ):

        stripped = code.rstrip()

        if (
            stripped.count('"') % 2
            == 1
        ):

            return (
                stripped
                + '"'
            )

        if (
            stripped.count("'") % 2
            == 1
        ):

            return (
                stripped
                + "'"
            )

    return None


# ============================================================
# NONE COMPARISON
# ============================================================

def _fix_none_comparison(
    code
):

    fixed = (
        code
        .replace(
            "== None",
            "is None"
        )
        .replace(
            "!= None",
            "is not None"
        )
    )

    if fixed != code:
        return fixed

    return None


# ============================================================
# JAVASCRIPT LOOSE EQUALITY
# ============================================================

def _fix_loose_equality(
    code
):

    placeholder = (
        "__NOT_EQUAL__"
    )

    fixed = (
        code
        .replace(
            "!==",
            placeholder
        )
        .replace(
            "!=",
            "!=="
        )
        .replace(
            "==",
            "==="
        )
        .replace(
            placeholder,
            "!=="
        )
    )

    if fixed != code:
        return fixed

    return None


# ============================================================
# VAR DECLARATION
# ============================================================

def _fix_var_declaration(
    code
):

    fixed = code.replace(
        "var ",
        "let ",
        1
    )

    if fixed != code:
        return fixed

    return None


# ============================================================
# BARE EXCEPT
# ============================================================

def _fix_bare_except(
    code
):

    fixed = code.replace(
        "except:",
        "except Exception as e:",
        1
    )

    if fixed != code:
        return fixed

    return None


# ============================================================
# MUTABLE DEFAULT ARGUMENT
# ============================================================

def _fix_mutable_default(
    code
):

    lines = code.splitlines()

    if not lines:
        return None

    match = re.search(
        r"(\w+)\s*=\s*(\[\]|\{\}|set\(\))",
        lines[0]
    )

    if not match:
        return None

    name = match.group(1)

    default_value = (
        match.group(2)
    )

    lines[0] = (
        lines[0][:match.start()]
        + f"{name}=None"
        + lines[0][match.end():]
    )

    indent = "    "

    if len(lines) > 1:

        whitespace = (
            lines[1]
            [
                :len(lines[1])
                - len(
                    lines[1].lstrip()
                )
            ]
        )

        if whitespace:
            indent = whitespace

    lines.insert(
        1,
        f"{indent}if {name} is None:"
    )

    lines.insert(
        2,
        f"{indent}    {name} = {default_value}"
    )

    return "\n".join(
        lines
    )


# ============================================================
# EMPTY CATCH BLOCK
# ============================================================

def _fix_empty_catch(
    code,
    file_path
):

    extension = os.path.splitext(
        file_path or ""
    )[1].lower()

    # catch(error) {}
    match = re.search(
        r"catch\s*\(([^)]*)\)\s*\{\s*\}",
        code
    )

    if match:

        parameters = (
            match.group(1).strip()
        )

        variable_match = re.search(
            r"([A-Za-z_$][\w$]*)\s*$",
            parameters
        )

        variable = (
            variable_match.group(1)
            if variable_match
            else "error"
        )

        if extension in {
            ".js",
            ".jsx",
            ".ts",
            ".tsx"
        }:

            statement = (
                f"console.error({variable});"
            )

        elif extension == ".java":

            statement = (
                f"{variable}.printStackTrace();"
            )

        elif extension == ".cs":

            statement = (
                f"Console.WriteLine({variable});"
            )

        else:
            return None

        replacement = (
            f"catch ({parameters}) "
            f"{{ {statement} }}"
        )

        return (
            code[:match.start()]
            + replacement
            + code[match.end():]
        )

    # catch {}
    match = re.search(
        r"catch\s*\{\s*\}",
        code
    )

    if match:

        if extension in {
            ".js",
            ".jsx",
            ".ts",
            ".tsx"
        }:

            replacement = (
                "catch (error) "
                "{ console.error(error); }"
            )

        else:
            return None

        return (
            code[:match.start()]
            + replacement
            + code[match.end():]
        )

    return None


# ============================================================
# REMOVE REPORTED LINE
# ============================================================

def _remove_reported_line(
    code,
    finding
):

    lines = code.splitlines()

    if not lines:
        return None

    try:

        target_line = int(
            finding.get(
                "line_start"
            )
        )

    except (
        TypeError,
        ValueError
    ):

        target_line = None

    try:

        chunk_start = int(
            finding.get(
                "chunk_line_start"
            )
        )

    except (
        TypeError,
        ValueError
    ):

        chunk_start = None

    if (
        target_line is not None
        and chunk_start is not None
    ):

        index = (
            target_line
            - chunk_start
        )

        if (
            0 <= index
            < len(lines)
        ):

            del lines[index]

            return "\n".join(
                lines
            )

    current_code = str(
        finding.get(
            "current_code",
            ""
        ) or ""
    ).strip()

    if current_code:

        for index, line in enumerate(
            lines
        ):

            if line.strip() == current_code:

                del lines[index]

                return "\n".join(
                    lines
                )

    return None


# ============================================================
# DIVISION BY ZERO
# ============================================================

def _fix_division_by_zero(
    code
):
    if not code or not str(code).strip():
        return None

    text = str(code)

    # Literal division by 0 or 0.0 → replace with a safe guarded form
    if re.search(r"/\s*0(\.0+)?\b", text):
        lines = text.splitlines()
        if len(lines) == 1:
            return (
                "# Division by zero is not allowed\n"
                f"# Original (broken): {text.strip()}\n"
                "raise ZeroDivisionError('Cannot divide by zero')"
            )
        return "\n".join(
            ["# Fixed: do not divide by zero"]
            + ["# " + line for line in lines]
            + ["raise ZeroDivisionError('Cannot divide by zero')"]
        )

    # Variable denominator → guard it
    match = re.search(
        r"/\s*([A-Za-z_]\w*)",
        text
    )
    if not match:
        return None

    denominator = match.group(1)
    lines = text.splitlines()

    if len(lines) == 1:
        return (
            f"if {denominator} != 0:\n"
            f"    {text}"
        )

    return "\n".join(
        [f"if {denominator} != 0:"]
        + ["    " + line for line in lines]
    )


# ============================================================
# DETERMINISTIC REPAIR ENGINE
# ============================================================

def _deterministic_repair(
    finding
):

    rule = str(
        finding.get(
            "rule",
            ""
        )
    ).lower()

    code = _get_code_context(
        finding
    )

    if not code:
        return None

    error_text = (
        f"{finding.get('error', '')} "
        f"{finding.get('cause', '')}"
    ).lower()

    # --------------------------------------------------------
    # SYNTAX ERROR
    # --------------------------------------------------------

    if (
        rule == "syntax_error"
        or "syntax error" in error_text
        or "indent" in error_text
        or "expected an indented block"
        in error_text
    ):

        fixed = _fix_python_syntax_error(
            finding
        )

        if (
            fixed
            and fixed != code
        ):

            return fixed

    # --------------------------------------------------------
    # NONE COMPARISON
    # --------------------------------------------------------

    if rule == "eq_none":

        fixed = _fix_none_comparison(
            code
        )

        if fixed:
            return fixed

    # --------------------------------------------------------
    # LOOSE EQUALITY
    # --------------------------------------------------------

    if rule == "loose_equality":

        fixed = _fix_loose_equality(
            code
        )

        if fixed:
            return fixed

    # --------------------------------------------------------
    # VAR
    # --------------------------------------------------------

    if rule == "var_declaration":

        fixed = _fix_var_declaration(
            code
        )

        if fixed:
            return fixed

    # --------------------------------------------------------
    # BARE EXCEPT
    # --------------------------------------------------------

    if rule == "bare_except":

        fixed = _fix_bare_except(
            code
        )

        if fixed:
            return fixed

    # --------------------------------------------------------
    # MUTABLE DEFAULT
    # --------------------------------------------------------

    if rule == "mutable_default_arg":

        fixed = _fix_mutable_default(
            code
        )

        if fixed:
            return fixed

    # --------------------------------------------------------
    # EMPTY CATCH
    # --------------------------------------------------------

    if rule == "empty_catch_block":

        fixed = _fix_empty_catch(
            code,
            finding.get(
                "file",
                ""
            )
        )

        if fixed:
            return fixed

    # --------------------------------------------------------
    # DIVISION BY ZERO
    # --------------------------------------------------------

    if rule == "possible_division_by_zero":

        fixed = _fix_division_by_zero(
            code
        )

        if fixed:
            return fixed

    # --------------------------------------------------------
    # UNNECESSARY / DEBUG CODE
    # --------------------------------------------------------

    if rule in {
        "leftover_console_statement",
        "leftover_debugger_statement",
        "leftover_debug_print",
        "unreachable_code"
    }:

        fixed = _remove_reported_line(
            code,
            finding
        )

        if fixed is not None:
            return fixed

    return None


# ============================================================
# FALLBACK REPORT
# ============================================================

def _snippet_for_text(code: str, max_len: int = 90) -> str:
    """Return a short, clean snippet of the buggy code for explanations."""
    if not code or not str(code).strip():
        return "this line of code"
    text = " ".join(str(code).strip().split())
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."
    return f"`{text}`"


def _fallback_cause_for_rule(rule: str, bug_type: str, code: str = "") -> str:
    """
    Build a UNIQUE, very simple cause that mentions the actual code.
    Every finding gets different text because the code snippet is different.
    """
    rule_key = str(rule or "").strip().lower()
    snippet = _snippet_for_text(code)

    if rule_key == "eq_none":
        return (
            f"Look at this code: {snippet}. "
            f"It is checking something with == None or != None. "
            f"In Python the safe way to ask 'is this empty / missing?' is to use "
            f"'is None' or 'is not None'. Using == can give the wrong answer if "
            f"the object has a special equality method."
        )
    if rule_key == "loose_equality":
        return (
            f"Look at this code: {snippet}. "
            f"It uses == which can secretly change the type of the values before comparing. "
            f"That means two different things can look the same by accident and the program "
            f"takes the wrong path."
        )
    if rule_key == "var_declaration":
        return (
            f"Look at this code: {snippet}. "
            f"It creates a variable with 'var'. 'var' is not limited to the small block "
            f"where it is written, so the same name can leak into other parts of the function "
            f"and cause surprising bugs."
        )
    if rule_key == "bare_except":
        return (
            f"Look at this code: {snippet}. "
            f"It says 'except:' with nothing after it. That means the program will catch "
            f"EVERY error, even serious ones that should stop the program. Real problems "
            f"can then be hidden and the program continues in a broken state."
        )
    if rule_key == "unreachable_code":
        return (
            f"Look at this code: {snippet}. "
            f"It sits right after a return, raise, break or continue. Once the program "
            f"hits that earlier statement it leaves the block, so this line can never run. "
            f"It is dead code that only confuses people who read the file."
        )
    if rule_key in {
        "leftover_console_statement",
        "leftover_debug_print",
        "leftover_debugger_statement",
    }:
        return (
            f"Look at this code: {snippet}. "
            f"This is a debug print / console / debugger left behind by a developer. "
            f"It should not be in the finished program because it prints private information "
            f"or slows the program down for no good reason."
        )
    if rule_key in {
        "syntax_error",
        "unclosed_bracket",
        "mismatched_bracket",
        "unexpected_closing_bracket",
        "unterminated_string",
    }:
        return (
            f"Look at this code: {snippet}. "
            f"The computer cannot even read this line properly. There is a missing quote, "
            f"a missing bracket, a wrong indent, or some other typing mistake. Until this "
            f"is fixed the whole file may refuse to run."
        )
    if rule_key == "mutable_default_arg":
        return (
            f"Look at this code: {snippet}. "
            f"A list or dictionary is written as the default value of a function argument. "
            f"Python creates that list only once, so every later call keeps changing the "
            f"same shared list. That is almost never what the programmer wanted."
        )
    if rule_key == "empty_catch_block":
        return (
            f"Look at this code: {snippet}. "
            f"An error is caught but the catch block does nothing with it. The program "
            f"quietly continues as if nothing went wrong, so real bugs stay hidden."
        )
    if rule_key == "possible_division_by_zero":
        return (
            f"Look at this code: {snippet}. "
            f"It divides by a value that might be zero. When the bottom number is zero "
            f"the computer raises a ZeroDivisionError and the program crashes."
        )
    if rule_key == "llm_file_analysis":
        return (
            f"Look at this code: {snippet}. "
            f"The AI scanner found a real problem here. The code as written can behave "
            f"incorrectly or crash when it runs."
        )

    # Generic but still unique because of the snippet
    return (
        f"Look at this code: {snippet}. "
        f"The scanner found a possible problem of type '{bug_type or 'Other'}'. "
        f"The way this line is written can cause the program to give wrong answers "
        f"or stop working."
    )


def _fallback_solution_for_rule(rule: str, code: str = "") -> str:
    """
    Build a UNIQUE, very simple solution that mentions the actual code.
    """
    rule_key = str(rule or "").strip().lower()
    snippet = _snippet_for_text(code)

    if rule_key == "eq_none":
        return (
            f"Change the comparison in {snippet} from '== None' to 'is None' "
            f"(or from '!= None' to 'is not None'). "
            f"That is the correct Python way to ask whether a value is missing."
        )
    if rule_key == "loose_equality":
        return (
            f"Change the '==' in {snippet} to '===' (or '!=' to '!=='). "
            f"Strict comparison does not secretly change types, so the check stays honest."
        )
    if rule_key == "var_declaration":
        return (
            f"Replace the word 'var' in {snippet} with 'let' or 'const'. "
            f"'let' and 'const' stay inside the block where they are written and do not leak."
        )
    if rule_key == "bare_except":
        return (
            f"Replace the bare 'except:' in {snippet} with a specific error type, "
            f"for example 'except ValueError as e:'. Then handle that error or re-raise it. "
            f"Do not catch everything silently."
        )
    if rule_key == "unreachable_code":
        return (
            f"Delete the line shown in {snippet}, or move it above the return/raise/break "
            f"so it can actually run. Dead code only confuses future readers."
        )
    if rule_key in {
        "leftover_console_statement",
        "leftover_debug_print",
        "leftover_debugger_statement",
    }:
        return (
            f"Delete the debug line shown in {snippet}. "
            f"If you still need logging, use a proper logger that can be turned off in production."
        )
    if rule_key in {
        "syntax_error",
        "unclosed_bracket",
        "mismatched_bracket",
        "unexpected_closing_bracket",
        "unterminated_string",
    }:
        return (
            f"Fix the typing mistake in {snippet}. "
            f"Make sure every quote and every bracket is opened and closed correctly, "
            f"and that the indentation matches the surrounding code. "
            f"Use the corrected code shown below."
        )
    if rule_key == "mutable_default_arg":
        return (
            f"Change the default in {snippet} to None. Inside the function write "
            f"'if the_argument is None: the_argument = []' (or {{}}). "
            f"Then each call gets its own fresh list or dictionary."
        )
    if rule_key == "empty_catch_block":
        return (
            f"Inside the empty catch block shown in {snippet}, at least print or log the error, "
            f"or re-raise it. Never swallow errors silently."
        )
    if rule_key == "possible_division_by_zero":
        return (
            f"Before the division in {snippet}, add a check such as "
            f"'if the bottom number is 0: return early or raise a clear error'. "
            f"Only divide when the bottom number is safe."
        )
    if rule_key == "llm_file_analysis":
        return (
            f"Replace the buggy code shown in {snippet} with the corrected version "
            f"in the box below. Read the corrected code carefully and paste it in place "
            f"of the old lines."
        )

    return (
        f"Replace the code shown in {snippet} with the corrected version in the box below. "
        f"That version removes the problem the scanner found."
    )


def _fallback_report(
    finding
):

    code = _get_code_context(
        finding
    )

    fixed = _deterministic_repair(
        finding
    )

    if not fixed:
        fixed = code

    bug_type = (
        finding.get(
            "bug_type"
        )
        or "Other"
    )

    if bug_type not in ALLOWED_BUG_TYPES:
        bug_type = "Other"

    rule = str(
        finding.get(
            "rule",
            ""
        )
    )

    confidence_map = {

        "eq_none": 96,

        "loose_equality": 96,

        "var_declaration": 92,

        "bare_except": 90,

        "unreachable_code": 96,

        "leftover_debugger_statement": 96,

        "leftover_console_statement": 94,

        "leftover_debug_print": 92,

        "syntax_error": 85,

        "unclosed_bracket": 85,

        "mismatched_bracket": 85,

        "unexpected_closing_bracket": 85,

        "unterminated_string": 85,

        "mutable_default_arg": 82,

        "empty_catch_block": 80,

        "possible_division_by_zero": 70
    }

    confidence = confidence_map.get(
        rule,
        45
    )

    repaired = (
        fixed
        and fixed.strip()
        != code.strip()
    )

    # Prefer any cause/solution already attached to the finding
    # (from the detector or from LLM file analysis). Only
    # fall back when they are missing.
    existing_cause = str(finding.get("cause") or "").strip()
    existing_solution = str(finding.get("solution") or "").strip()

    # Use the most specific code we have so the explanation
    # mentions the real buggy lines.
    code_for_text = (
        str(finding.get("current_code") or "").strip()
        or code
    )

    cause = existing_cause or _fallback_cause_for_rule(
        rule, bug_type, code_for_text
    )
    solution = existing_solution or _fallback_solution_for_rule(
        rule, code_for_text
    )

    return {

        "error": finding.get(
            "error",
            "Possible issue"
        ),

        "bug_type": bug_type,

        "cause": cause,

        "why_occurs": cause,

        "solution_type": "replace",

        "solution": solution,

        "replacement_code":
            _clean_replacement_code(
                fixed
            ),

        "add_location": None,

        "new_file_path": None,

        "explanation": (
            "The corrected-code section contains "
            "the best repair available for this exact line."
        ),

        "confidence": confidence,

        "insufficient_evidence":
            not repaired
    }


# ============================================================
# GUARANTEE CLEAN REPLACEMENT CODE
# ============================================================

def _ensure_replacement_code(
    result,
    finding
):

    if not isinstance(
        result,
        dict
    ):

        result = _fallback_report(
            finding
        )

    # --------------------------------------------------------
    # Detector replacement
    # --------------------------------------------------------

    detector_code = finding.get(
        "replacement_code"
    )

    if (
        isinstance(
            detector_code,
            str
        )
        and detector_code.strip()
    ):

        cleaned = (
            _clean_replacement_code(
                detector_code
            )
        )

        if cleaned:

            result[
                "replacement_code"
            ] = cleaned

            result[
                "solution_type"
            ] = "replace"

            result[
                "insufficient_evidence"
            ] = False

            return result

    # --------------------------------------------------------
    # Gemini replacement
    # --------------------------------------------------------

    replacement = result.get(
        "replacement_code"
    )

    if (
        isinstance(
            replacement,
            str
        )
        and replacement.strip()
    ):

        cleaned = (
            _clean_replacement_code(
                replacement
            )
        )

        if cleaned:

            result[
                "replacement_code"
            ] = cleaned

            return result

    # --------------------------------------------------------
    # Deterministic replacement
    # --------------------------------------------------------

    fixed = _deterministic_repair(
        finding
    )

    if fixed and fixed.strip():

        result[
            "replacement_code"
        ] = _clean_replacement_code(
            fixed
        )

        result[
            "solution_type"
        ] = "replace"

        result[
            "insufficient_evidence"
        ] = False

        return result

    # --------------------------------------------------------
    # Last fallback
    # --------------------------------------------------------

    result[
        "replacement_code"
    ] = _clean_replacement_code(
        _get_code_context(
            finding
        )
    )

    result[
        "solution_type"
    ] = "replace"

    result[
        "insufficient_evidence"
    ] = True

    return result


# ============================================================
# NORMALIZE RESULT
# ============================================================

def _normalize_result(
    result,
    finding
):

    result = _ensure_replacement_code(
        result,
        finding
    )

    bug_type = result.get(
        "bug_type"
    )

    if bug_type not in ALLOWED_BUG_TYPES:

        result[
            "bug_type"
        ] = (
            finding.get(
                "bug_type"
            )
            or "Other"
        )

    try:

        confidence = int(
            result.get(
                "confidence",
                70
            )
        )

    except (
        TypeError,
        ValueError
    ):

        confidence = 70

    confidence = max(
        0,
        min(
            100,
            confidence
        )
    )

    result[
        "confidence"
    ] = confidence

    insufficient = bool(
        result.get(
            "insufficient_evidence",
            False
        )
    )

    if (
        confidence >= 70
        and not insufficient
    ):

        result[
            "confidence_level"
        ] = "High Confidence"

    else:

        result[
            "confidence_level"
        ] = "Low Confidence"

    code_for_text = str(
        finding.get("current_code")
        or finding.get("code_chunk")
        or ""
    ).strip()

    if not str(
        result.get(
            "solution",
            ""
        ) or ""
    ).strip():

        result[
            "solution"
        ] = _fallback_solution_for_rule(
            str(
                finding.get(
                    "rule",
                    ""
                )
            ),
            code_for_text,
        )

    if not str(
        result.get(
            "cause",
            ""
        ) or ""
    ).strip():

        result[
            "cause"
        ] = _fallback_cause_for_rule(
            str(
                finding.get(
                    "rule",
                    ""
                )
            ),
            str(
                result.get(
                    "bug_type",
                    finding.get(
                        "bug_type",
                        "Other",
                    ),
                )
            ),
            code_for_text,
        )

        result[
            "why_occurs"
        ] = result["cause"]

    # Final safety cleanup.
    result[
        "replacement_code"
    ] = _clean_replacement_code(
        result.get(
            "replacement_code",
            ""
        )
    )

    return result


# ============================================================
# MAIN LLM BUG ANALYSIS
# ============================================================

def analyze_finding(
    finding,
    retrieved
):

    keys = _get_api_keys()

    # --------------------------------------------------------
    # NO API KEY
    # --------------------------------------------------------

    if not keys:

        print(
            "[llm_client] No Gemini API key found. "
            "Using local repair engine."
        )

        return _normalize_result(
            _fallback_report(
                finding
            ),
            finding
        )

    payload = {

        "system_instruction": {
            "parts": [
                {
                    "text":
                    SYSTEM_PROMPT
                }
            ]
        },

        "contents": [
            {
                "role": "user",

                "parts": [
                    {
                        "text":
                        _build_user_message(
                            finding,
                            retrieved
                        )
                    }
                ]
            }
        ],

        "generationConfig": {

            "response_mime_type":
                "application/json",

            "maxOutputTokens":
                3000
        }
    }

    last_error = None

    rate_limited_response = None

    for number, api_key in enumerate(
        keys,
        start=1
    ):

        try:

            response = requests.post(

                GEMINI_URL,

                params={
                    "key": api_key
                },

                json=payload,

                timeout=
                GEMINI_REQUEST_TIMEOUT_SECONDS
            )

            # ------------------------------------------------
            # SUCCESS
            # ------------------------------------------------

            if response.status_code == 200:

                data = response.json()

                candidates = data.get(
                    "candidates",
                    []
                )

                if not candidates:

                    raise ValueError(
                        "Gemini returned no candidates."
                    )

                parts = (
                    candidates[0]
                    .get(
                        "content",
                        {}
                    )
                    .get(
                        "parts",
                        []
                    )
                )

                if not parts:

                    raise ValueError(
                        "Gemini returned no content."
                    )

                raw = parts[0].get(
                    "text",
                    ""
                )

                result = json.loads(
                    _clean_json(
                        raw
                    )
                )

                return _normalize_result(
                    result,
                    finding
                )

            # ------------------------------------------------
            # 429 QUOTA
            # ------------------------------------------------

            if response.status_code == 429:

                rate_limited_response = (
                    response
                )

                print(
                    "[llm_client] Gemini quota "
                    "exceeded. Using local repair."
                )

                break

            # ------------------------------------------------
            # 503
            # ------------------------------------------------

            if response.status_code == 503:

                last_error = (
                    "Gemini temporarily unavailable."
                )

                print(
                    "[llm_client] Gemini temporarily "
                    "unavailable. Using local repair."
                )

                break

            # ------------------------------------------------
            # OTHER HTTP ERROR
            # ------------------------------------------------

            last_error = (
                f"Gemini key {number} "
                f"returned HTTP "
                f"{response.status_code}"
            )

        except Exception as exc:

            last_error = str(
                exc
            )

            print(
                f"[llm_client] Gemini request "
                f"failed: {exc}"
            )

    # ========================================================
    # GEMINI FAILED
    # ========================================================

    result = _normalize_result(
        _fallback_report(
            finding
        ),
        finding
    )

    if rate_limited_response is not None:

        retry_seconds, is_daily = (
            _parse_rate_limit_info(
                rate_limited_response
            )
        )

        result[
            "rate_limited"
        ] = True

        result[
            "rate_limit_message"
        ] = _build_rate_limit_message(
            retry_seconds,
            is_daily
        )

    elif last_error:

        result[
            "llm_error"
        ] = last_error

    return result


# ============================================================
# FILE ANALYSIS
# ============================================================

def analyze_file(
    file_path,
    source
):

    if (
        not source
        or not source.strip()
    ):

        return []

    keys = _get_api_keys()

    if not keys:
        return []

    prompt = f"""
You are a precise software bug detector and repair engine.

Review the source file below and report EVERY real, evidence-based bug you can prove from the code itself.
Do NOT invent bugs. Do NOT report style preferences or pure opinions. Only report issues that would cause
incorrect behaviour, a crash, a syntax error, a security problem, or clearly dead/unnecessary code.

File path: {file_path}

For EACH bug return one JSON object with ALL of these fields:

- error: short title of the bug (e.g. "Missing closing parenthesis", "Division by zero risk")
- bug_type: one of Runtime Error, Logic Error, Syntax Error, Type Error, Dependency Error, Security Issue, Performance Issue, API Error, Unnecessary Code, Other
- rule: a short machine-friendly tag (e.g. syntax_error, possible_division_by_zero, bare_except)
- line_start: 1-based start line
- line_end: 1-based end line
- function: name of the enclosing function/method if known, otherwise null
- current_code: the exact buggy code snippet (copy from the file)
- cause: Write a short paragraph in VERY SIMPLE everyday English that a 5-year-old could understand.
  Explain what is wrong, where it is wrong, and what bad thing can happen. Mention real variable/function
  names from THIS code. Every cause must be unique to this exact bug and must NOT copy generic text.
- solution: Write a short paragraph in VERY SIMPLE everyday English that a 5-year-old could understand.
  Explain the exact steps to fix THIS bug. Every solution must be unique to this exact bug.
- replacement_code: the corrected source code only (no markdown, no labels, no explanations). This must
  be ready to copy-paste in place of current_code.
- confidence: integer 0-100

Return ONLY a valid JSON array of such objects. If the file has no real bugs, return [].

FILE:
--------------------------------
{source}
--------------------------------
"""

    payload = {

        "system_instruction": {
            "parts": [
                {
                    "text":
                    "You are a precise software "
                    "bug detector. Return JSON only."
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
                ]
            }
        ],

        "generationConfig": {

            "response_mime_type":
                "application/json",

            "maxOutputTokens":
                8192
        }
    }

    for number, api_key in enumerate(
        keys,
        start=1
    ):

        try:

            response = requests.post(

                GEMINI_URL,

                params={
                    "key": api_key
                },

                json=payload,

                timeout=
                GEMINI_REQUEST_TIMEOUT_SECONDS
            )

            if response.status_code in {
                429,
                503
            }:

                print(
                    "[llm_client] File analysis "
                    "Gemini unavailable. "
                    "Skipping LLM file analysis."
                )

                return []

            if response.status_code != 200:
                continue

            data = response.json()

            raw = (
                data[
                    "candidates"
                ][0]
                [
                    "content"
                ]
                [
                    "parts"
                ][0]
                [
                    "text"
                ]
            )

            parsed = json.loads(
                _clean_json(
                    raw
                )
            )

            if not isinstance(
                parsed,
                list
            ):

                return []

            findings = []

            for item in parsed:

                if (
                    not isinstance(
                        item,
                        dict
                    )
                    or not item.get(
                        "error"
                    )
                ):

                    continue

                bug_type = (
                    item.get(
                        "bug_type"
                    )
                    or "Other"
                )

                if (
                    bug_type
                    not in ALLOWED_BUG_TYPES
                ):

                    bug_type = "Other"

                try:

                    line_start = int(
                        item.get(
                            "line_start"
                        )
                    )

                except (
                    TypeError,
                    ValueError
                ):

                    line_start = None

                try:

                    line_end = int(
                        item.get(
                            "line_end"
                        )
                    )

                except (
                    TypeError,
                    ValueError
                ):

                    line_end = line_start

                rule = str(
                    item.get("rule") or "llm_file_analysis"
                ).strip() or "llm_file_analysis"

                replacement = _clean_replacement_code(
                    str(item.get("replacement_code") or "")
                )

                solution = str(
                    item.get("solution") or ""
                ).strip()

                try:
                    conf = int(item.get("confidence", 75))
                except (TypeError, ValueError):
                    conf = 75
                conf = max(0, min(100, conf))

                findings.append(
                    {
                        "error": str(item["error"]),
                        "bug_type": bug_type,
                        "cause": str(
                            item.get("cause") or ""
                        ),
                        "solution": solution,
                        "replacement_code": replacement,
                        "line_start": line_start,
                        "line_end": line_end,
                        "current_code": str(
                            item.get("current_code") or ""
                        ),
                        "function": item.get("function"),
                        "rule": rule,
                        "file": file_path,
                        "confidence": conf,
                    }
                )

            return findings

        except Exception as exc:

            print(
                f"[llm_client] File analysis "
                f"key {number} failed: "
                f"{exc}"
            )

    return []


# ============================================================
# PUBLIC FALLBACK
# ============================================================

def get_fallback_report(
    finding
):

    return _normalize_result(
        _fallback_report(
            finding
        ),
        finding
    )