"""
LLM client for RAG GitHub Bug Detection and Recovery.

Used by main.py:
    analyze_finding()
    analyze_file()
"""

import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv


load_dotenv(
    dotenv_path=Path(__file__).resolve().with_name(".env"),
    override=False,
)


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

GEMINI_MODEL = "gemini-3.8-flash"

GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)

LLM_UNAVAILABLE_MESSAGE = (
    "LLM limit completed / unavailable\n"
    "The cause and solution could not be generated because the LLM is "
    "currently unavailable or its usage limit has been reached. Please "
    "try again later."
)


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
    numbered_names = sorted(
        (
            name
            for name in os.environ
            if re.fullmatch(r"GEMINI_API_KEY_\d+", name)
        ),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    for name in numbered_names:
        key = os.getenv(name, "").strip()
        if key and key not in keys:
            keys.append(key)

    key = os.getenv(
        "GEMINI_API_KEY"
    )

    if key and key.strip():

        key = key.strip()

        if key not in keys:
            keys.append(key)

    return keys


GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_MAX_TOKENS = 6000   # generous budget for gpt-oss-20b reasoning model
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def _get_groq_api_keys() -> List[str]:
    keys = []
    numbered_names = sorted(
        (
            name
            for name in os.environ
            if re.fullmatch(r"GROQ_API_KEY_\d+", name)
        ),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    for name in numbered_names:
        key = os.getenv(name, "").strip()
        if key and key not in keys:
            keys.append(key)

    # Also support a single legacy Groq key.
    key = os.getenv("GROQ_API_KEY")
    if key and key.strip() and key.strip() not in keys:
        keys.append(key.strip())

    return keys


_PROVIDER_ROTATION_LOCK = threading.Lock()
_PROVIDER_ROTATION_INDEX = 0


def _get_provider_credentials():
    """Return configured credentials in a round-robin Gemini/Groq order."""
    global _PROVIDER_ROTATION_INDEX

    gemini_keys = _get_api_keys()
    groq_keys = _get_groq_api_keys()
    credentials = []

    for index in range(max(len(gemini_keys), len(groq_keys))):
        if index < len(gemini_keys):
            credentials.append(("gemini", index + 1, gemini_keys[index]))
        if index < len(groq_keys):
            credentials.append(("groq", index + 1, groq_keys[index]))

    if not credentials:
        return []

    with _PROVIDER_ROTATION_LOCK:
        start = _PROVIDER_ROTATION_INDEX % len(credentials)
        _PROVIDER_ROTATION_INDEX = (start + 1) % len(credentials)

    return credentials[start:] + credentials[:start]


def _groq_request(api_key, system_prompt, user_prompt, max_tokens=None):
    body = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
    }
    # Only set max_completion_tokens when explicitly requested.
    # Leaving it unset lets Groq use its full output capacity.
    if max_tokens is not None:
        body["max_completion_tokens"] = max_tokens

    response = requests.post(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=120,
    )

    if response.status_code == 429:
        raise RuntimeError("Groq rate-limited (HTTP 429)")

    if response.status_code != 200:
        raise RuntimeError(
            f"Groq HTTP {response.status_code}: {response.text[:300]}"
        )

    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("Groq returned no choices")

    message = choices[0].get("message", {})
    content = message.get("content")

    # gpt-oss-20b is a reasoning model — final answer is in `content`.
    # If content is empty (can happen when finish_reason=length),
    # fall back to extracting the JSON block from `reasoning`.
    if not isinstance(content, str) or not content.strip():
        reasoning = message.get("reasoning") or ""
        match = re.search(r"\{[\s\S]+\}", reasoning)
        if match:
            content = match.group(0)
        else:
            finish = choices[0].get("finish_reason", "")
            raise ValueError(
                f"Groq returned empty content (finish_reason={finish})"
            )

    return content

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
   - Analyze only the CURRENT BUG and its supplied code chunk. Do not
     combine it with another finding or reuse another finding's answer.
   - Explain what is wrong, where the problem happens, and why it happens.
   - Use very simple, everyday English that a beginner or a 5-year-old
     could understand.
   - Write a natural paragraph. Use as many sentences as needed to explain
     the problem clearly.
   - Mention the actual variable, condition, function, or operation when
     relevant.
   - Explain what could happen when this code runs.
   - Make the explanation specific to this exact code cell.
   - Different code cells must receive independently written explanations
     based on their own code, even when they have the same bug type.
   - Avoid difficult technical words. If a technical word is necessary,
     explain what it means in simple English.
   - Never guess or invent details that are not supported by the code.

2. solution:
   - Explain how to fix the exact problem found in this code.
   - Base the fix on this finding's actual code, not only its bug type or
     a similar historical example.
   - Use very simple English that a beginner or a 5-year-old could understand.
   - Write a natural paragraph and use as many sentences as needed.
   - Explain what needs to change, where to change it, and why the change
     will solve the problem.
   - Give clear steps in words, not vague advice such as "fix the error"
     or "check the code".
   - Make the solution specific to this code cell.
   - Do not reuse a generic solution just because another finding has
     the same bug type.
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

    match = re.search(
        r"/\s*([A-Za-z_]\w*)",
        code
    )

    if not match:
        return None

    denominator = (
        match.group(1)
    )

    lines = code.splitlines()

    if len(lines) == 1:

        return (
            f"if {denominator} != 0:\n"
            f"    {code}"
        )

    return "\n".join(
        [
            f"if {denominator} != 0:"
        ]
        + [
            "    " + line
            for line in lines
        ]
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
        return get_llm_unavailable_report(finding)

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


def get_llm_unavailable_report(finding, error=None):
    """
    Return a fallback report when all LLM providers fail.
    Always includes the detected buggy code so the frontend can show it.
    """
    # Use deterministic repair as best-effort replacement code
    det_code = _deterministic_repair(finding) if finding else None
    if det_code:
        det_code = _clean_replacement_code(det_code)

    current_code = str(finding.get("current_code", "") or "") if finding else ""

    result = {
        "error": finding.get("error", "Detected issue") if finding else "Detected issue",
        "bug_type": finding.get("bug_type", "Other") if finding else "Other",
        "cause": "",
        "why_occurs": "",
        "solution": "",
        # Always put something in replacement_code — at minimum the raw detected code
        "replacement_code": det_code or _clean_replacement_code(current_code) or None,
        "current_code": current_code,
        "confidence": 0,
        "confidence_level": "Low Confidence",
        "insufficient_evidence": True,
        "llm_unavailable": True,
        "llm_unavailable_notice": LLM_UNAVAILABLE_MESSAGE,
        "analysis_source": "unavailable",
    }
    if error:
        result["llm_error"] = str(error)
    return result

def _analyze_finding_with_groq(finding, retrieved, keys=None):
    if keys is None:
        keys = _get_groq_api_keys()
    if not keys:
        return None

    prompt = _build_user_message(finding, retrieved)
    last_error = None

    for number, api_key in enumerate(keys, start=1):
        try:
            raw = _groq_request(
                api_key=api_key,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt,
            )

            result = json.loads(_clean_json(raw))

            if not isinstance(result, dict):
                raise ValueError("Groq response is not a JSON object")

            if not all(
                isinstance(result.get(field), str)
                and result[field].strip()
                for field in ("cause", "solution")
            ):
                raise ValueError("Groq response is missing cause or solution")

            result = _normalize_result(result, finding)
            result["analysis_source"] = "groq"
            return result

        except Exception as exc:
            last_error = str(exc)
            print(
                f"[llm_client] Groq key {number} failed: {exc}"
            )

    print(f"[llm_client] All Groq keys failed: {last_error}")
    return None


# ============================================================
# MAIN LLM BUG ANALYSIS
# ============================================================

def _analyze_file_with_groq(file_path, source):
    if not source or not source.strip():
        return []

    prompt = f"""
Review this source file and report only real, evidence-based bugs.

File: {file_path}

Return ONLY a JSON array. Each object must contain:
error, bug_type, cause, line_start, line_end, current_code, function

Use 1-based line numbers. Do not invent bugs.

FILE:
--------------------------------
{source}
--------------------------------
"""

    for number, api_key in enumerate(_get_groq_api_keys(), start=1):
        try:
            raw = _groq_request(
                api_key=api_key,
                system_prompt=(
                    "You are a precise software bug detector. "
                    "Return JSON only. Report evidence-based bugs."
                ),
                user_prompt=prompt,
                max_tokens=2500,
            )

            parsed = json.loads(_clean_json(raw))
            if not isinstance(parsed, list):
                raise ValueError("Expected a JSON array")

            findings = []

            for item in parsed:
                if not isinstance(item, dict) or not item.get("error"):
                    continue

                bug_type = item.get("bug_type") or "Other"
                if bug_type not in ALLOWED_BUG_TYPES:
                    bug_type = "Other"

                try:
                    line_start = int(item.get("line_start"))
                except (TypeError, ValueError):
                    line_start = None

                try:
                    line_end = int(item.get("line_end"))
                except (TypeError, ValueError):
                    line_end = line_start

                findings.append({
                    "error": str(item["error"]),
                    "bug_type": bug_type,
                    "cause": str(item.get("cause") or ""),
                    "line_start": line_start,
                    "line_end": line_end,
                    "current_code": str(item.get("current_code") or ""),
                    "function": item.get("function"),
                    "rule": "llm_file_analysis",
                    "file": file_path,
                })

            return findings

        except Exception as exc:
            print(f"[llm_client] Groq file-analysis key {number} failed: {exc}")

    return []

def analyze_finding(
    finding,
    retrieved
):
    """
    Exhaust every available Gemini and Groq key to get a real LLM analysis.

    Strategy:
    - Try all keys (Gemini + Groq) in interleaved round-robin order.
    - On rate-limit (429): note it, keep trying other keys.
    - If every key on this pass was rate-limited: wait and retry up to
      MAX_FULL_PASSES times — no artificial early exit.
    - Only return llm_unavailable when every key on every pass has
      permanently failed (not just rate-limited).
    - No restriction on token output — use the full quota each provider allows.
    """
    import time

    # How many full passes through all credentials before giving up.
    # 5 passes * (15 Gemini + 4 Groq) = up to 95 attempts per bug.
    MAX_FULL_PASSES = 5
    BASE_RATE_LIMIT_WAIT = 3    # seconds; multiplied by pass number (3,6,9,12)

    user_message = _build_user_message(finding, retrieved)
    gemini_payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": user_message}]}],
        "generationConfig": {
            "response_mime_type": "application/json",
            # No maxOutputTokens restriction — let Gemini use its full capacity
        },
    }

    last_error = None

    for pass_num in range(1, MAX_FULL_PASSES + 1):
        # Re-fetch credentials each pass so rotation distributes across bugs
        all_credentials = _get_provider_credentials()
        if not all_credentials:
            return get_llm_unavailable_report(finding)

        rate_limited_count = 0
        total_count = len(all_credentials)

        for provider, number, api_key in all_credentials:
            try:
                # ── GROQ ──────────────────────────────────────────
                if provider == "groq":
                    try:
                        prompt = _build_user_message(finding, retrieved)
                        raw = _groq_request(
                            api_key=api_key,
                            system_prompt=SYSTEM_PROMPT,
                            user_prompt=prompt,
                            # No token cap — use Groq's full output limit
                            max_tokens=None,
                        )
                        result = json.loads(_clean_json(raw))
                        if not isinstance(result, dict):
                            raise ValueError("Groq response is not a JSON object")
                        if not all(
                            isinstance(result.get(f), str) and result[f].strip()
                            for f in ("cause", "solution")
                        ):
                            raise ValueError("Groq response missing cause or solution")
                        result = _normalize_result(result, finding)
                        result["analysis_source"] = "groq"
                        print(f"[llm_client] Groq key {number} succeeded (pass {pass_num})")
                        return result
                    except RuntimeError as exc:
                        if "429" in str(exc):
                            rate_limited_count += 1
                            print(f"[llm_client] Groq key {number} rate-limited (pass {pass_num})")
                        else:
                            last_error = str(exc)
                            print(f"[llm_client] Groq key {number} failed: {exc}")
                    except Exception as exc:
                        last_error = str(exc)
                        print(f"[llm_client] Groq key {number} error: {exc}")
                    continue

                # ── GEMINI ────────────────────────────────────────
                response = requests.post(
                    GEMINI_URL,
                    params={"key": api_key},
                    json=gemini_payload,
                    timeout=90,
                )

                if response.status_code == 429:
                    rate_limited_count += 1
                    print(
                        f"[llm_client] Gemini key {number} rate-limited "
                        f"(pass {pass_num}/{MAX_FULL_PASSES})"
                    )
                    continue

                if response.status_code == 503:
                    last_error = f"Gemini key {number} temporarily unavailable (503)"
                    print(f"[llm_client] {last_error}")
                    continue

                if response.status_code != 200:
                    last_error = (
                        f"Gemini key {number} HTTP {response.status_code}: "
                        f"{response.text[:200]}"
                    )
                    print(f"[llm_client] {last_error}")
                    continue

                data = response.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    # RECITATION or SAFETY block — not a quota issue
                    finish = (candidates[0].get("finishReason") if candidates else
                              data.get("candidates", [{}])[0].get("finishReason", ""))
                    last_error = f"Gemini key {number} returned no usable candidates (finishReason={finish})"
                    print(f"[llm_client] {last_error}")
                    continue

                parts = candidates[0].get("content", {}).get("parts", [])
                if not parts:
                    last_error = f"Gemini key {number} returned no content parts"
                    print(f"[llm_client] {last_error}")
                    continue

                raw_text = parts[0].get("text", "")
                result = json.loads(_clean_json(raw_text))

                if not isinstance(result, dict):
                    raise ValueError("Gemini response was not a JSON object")
                if not all(
                    isinstance(result.get(f), str) and result[f].strip()
                    for f in ("cause", "solution")
                ):
                    raise ValueError("Gemini did not provide both cause and solution")

                result = _normalize_result(result, finding)
                result["analysis_source"] = "gemini"
                print(f"[llm_client] Gemini key {number} succeeded (pass {pass_num})")
                return result

            except Exception as exc:
                last_error = (
                    f"{provider.title()} key {number} exception "
                    f"({type(exc).__name__}): {exc}"
                )
                print(f"[llm_client] {last_error}")

        # ── End of pass ───────────────────────────────────────────
        print(
            f"[llm_client] Pass {pass_num}/{MAX_FULL_PASSES} exhausted. "
            f"Rate-limited: {rate_limited_count}/{total_count}. "
            f"Last error: {last_error}"
        )

        if pass_num < MAX_FULL_PASSES:
            if rate_limited_count > 0:
                # At least some keys were rate-limited — wait for quota reset
                wait = BASE_RATE_LIMIT_WAIT * pass_num
                print(f"[llm_client] Waiting {wait}s for rate limits to clear…")
                time.sleep(wait)
            else:
                # All permanent failures — short pause then retry
                time.sleep(3)

    print(f"[llm_client] All {MAX_FULL_PASSES} passes exhausted for this bug. Returning fallback.")
    return get_llm_unavailable_report(finding, error=last_error)


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
Review this source file and report only
real, evidence-based bugs.

File:
{file_path}

Return ONLY a JSON array.

Each object must contain:

error
bug_type
cause
line_start
line_end
current_code
function

Use 1-based line numbers.

Do not invent bugs.

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
                2500
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

                json=payload
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

                findings.append(
                    {

                        "error": str(
                            item["error"]
                        ),

                        "bug_type":
                            bug_type,

                        "cause": str(
                            item.get(
                                "cause"
                            ) or ""
                        ),

                        "line_start":
                            line_start,

                        "line_end":
                            line_end,

                        "current_code":
                            str(
                                item.get(
                                    "current_code"
                                ) or ""
                            ),

                        "function":
                            item.get(
                                "function"
                            ),

                        "rule":
                            "llm_file_analysis",

                        "file":
                            file_path
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
