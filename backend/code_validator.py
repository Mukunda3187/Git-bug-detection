import ast
import os
import re
import subprocess
import tempfile
from typing import Any, Dict


# ============================================================
# LANGUAGE DETECTION
# ============================================================

LANGUAGE_BY_EXTENSION = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".cs": "csharp",
    ".go": "go",
    ".php": "php",
}


# ============================================================
# BASIC STRUCTURAL VALIDATION
# ============================================================

def _validate_brackets_and_strings(code: str):
    """
    Basic structural validation.

    Does not execute the generated code.
    """

    if not code or not code.strip():
        return False, "Code is empty."

    stack = []

    pairs = {
        ")": "(",
        "]": "[",
        "}": "{",
    }

    opening = set(pairs.values())

    in_single = False
    in_double = False
    in_backtick = False
    escaped = False

    i = 0

    while i < len(code):
        char = code[i]

        if escaped:
            escaped = False
            i += 1
            continue

        if char == "\\":
            if in_single or in_double or in_backtick:
                escaped = True

            i += 1
            continue

        if char == "'" and not in_double and not in_backtick:
            in_single = not in_single
            i += 1
            continue

        if char == '"' and not in_single and not in_backtick:
            in_double = not in_double
            i += 1
            continue

        if char == "`" and not in_single and not in_double:
            in_backtick = not in_backtick
            i += 1
            continue

        if in_single or in_double or in_backtick:
            i += 1
            continue

        if char in opening:
            stack.append(char)

        elif char in pairs:
            if not stack:
                return False, f"Unexpected closing bracket '{char}'."

            if stack[-1] != pairs[char]:
                return False, f"Mismatched bracket '{char}'."

            stack.pop()

        i += 1

    if in_single:
        return False, "Unclosed single-quoted string."

    if in_double:
        return False, "Unclosed double-quoted string."

    if in_backtick:
        return False, "Unclosed backtick string."

    if stack:
        return False, "Unclosed bracket."

    return True, "Basic structural validation passed."


# ============================================================
# PYTHON VALIDATION
# ============================================================

def _validate_python(code: str):
    if not code or not code.strip():
        return False, "Python code is empty."

    try:
        ast.parse(code)
        return True, "Python syntax validation passed."

    except SyntaxError as exc:
        line = exc.lineno or "unknown"

        return (
            False,
            f"Python syntax error at line {line}: {exc.msg}",
        )

    except Exception as exc:
        return (
            False,
            f"Python validation failed: {exc}",
        )


# ============================================================
# JAVASCRIPT VALIDATION
# ============================================================

def _validate_javascript(code: str):
    if not code or not code.strip():
        return False, "JavaScript code is empty."

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".js",
            delete=False,
            encoding="utf-8",
        ) as temp_file:

            temp_file.write(code)
            temp_path = temp_file.name

        try:
            result = subprocess.run(
                [
                    "node",
                    "--check",
                    temp_path,
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                return (
                    True,
                    "JavaScript syntax validation passed.",
                )

            error_message = (
                result.stderr.strip()
                or result.stdout.strip()
                or "JavaScript syntax error."
            )

            return False, error_message

        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

    except (
        FileNotFoundError,
        subprocess.SubprocessError,
        OSError,
    ):
        return _validate_brackets_and_strings(code)


# ============================================================
# TYPESCRIPT VALIDATION
# ============================================================

def _validate_typescript(code: str):
    if not code or not code.strip():
        return False, "TypeScript code is empty."

    try:
        with tempfile.TemporaryDirectory() as temp_dir:

            file_path = os.path.join(
                temp_dir,
                "replacement.ts",
            )

            with open(
                file_path,
                "w",
                encoding="utf-8",
            ) as file:
                file.write(code)

            result = subprocess.run(
                [
                    "tsc",
                    "--noEmit",
                    file_path,
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )

            if result.returncode == 0:
                return (
                    True,
                    "TypeScript syntax validation passed.",
                )

            error_message = (
                result.stderr.strip()
                or result.stdout.strip()
                or "TypeScript validation failed."
            )

            return False, error_message

    except (
        FileNotFoundError,
        subprocess.SubprocessError,
        OSError,
    ):
        return _validate_brackets_and_strings(code)


# ============================================================
# GENERAL LANGUAGE VALIDATION
# ============================================================

def _validate_general_language(code: str, language: str):
    valid, message = _validate_brackets_and_strings(code)

    if not valid:
        return False, message

    return (
        True,
        f"{language} structural validation passed.",
    )


# ============================================================
# VALIDATE SOURCE CODE
# ============================================================

def _validate_code(file_path: str, code: str):
    """
    Validate a piece of source code according to its language.
    """

    if not code or not str(code).strip():
        return {
            "valid": False,
            "message": "Code is empty.",
            "language": "unknown",
            "validation_method": "none",
        }

    extension = os.path.splitext(
        str(file_path or "")
    )[1].lower()

    language = LANGUAGE_BY_EXTENSION.get(
        extension,
        "unknown",
    )

    if language == "python":
        valid, message = _validate_python(code)

        return {
            "valid": valid,
            "message": message,
            "language": language,
            "validation_method": "Python AST",
        }

    if language == "javascript":
        valid, message = _validate_javascript(code)

        return {
            "valid": valid,
            "message": message,
            "language": language,
            "validation_method": "Node.js / structural",
        }

    if language == "typescript":
        valid, message = _validate_typescript(code)

        return {
            "valid": valid,
            "message": message,
            "language": language,
            "validation_method": "TypeScript compiler / structural",
        }

    valid, message = _validate_general_language(
        code,
        language,
    )

    return {
        "valid": valid,
        "message": message,
        "language": language,
        "validation_method": "Structural validation",
    }


# ============================================================
# NORMALIZE ACTION
# ============================================================

def _normalize_solution_type(result: Dict[str, Any], finding: Dict[str, Any]):
    """
    Every finding must have one correction action:

        replace
        remove
        add
    """

    value = str(
        result.get("solution_type")
        or ""
    ).strip().lower()

    aliases = {
        "replacement": "replace",
        "replace_code": "replace",
        "delete": "remove",
        "remove_code": "remove",
        "insertion": "add",
        "insert": "add",
        "add_code": "add",
    }

    value = aliases.get(value, value)

    if value not in {
        "replace",
        "remove",
        "add",
    }:

        rule = str(
            finding.get("rule")
            or ""
        ).lower()

        if rule in {
            "leftover_console_statement",
            "leftover_debugger_statement",
            "leftover_debug_print",
            "unreachable_code",
        }:
            value = "remove"

        else:
            value = "replace"

    result["solution_type"] = value

    return value


# ============================================================
# FIND REPORTED LINE
# ============================================================

def _find_reported_line(
    original_code: str,
    finding: Dict[str, Any],
):
    """
    Find the line reported by the detector inside current_code.

    Returns:
        {
            "found": bool,
            "index": int | None,
            "line": str | None
        }
    """

    if not original_code:
        return {
            "found": False,
            "index": None,
            "line": None,
        }

    lines = str(original_code).splitlines()

    # --------------------------------------------------------
    # Try line number relative to chunk
    # --------------------------------------------------------

    try:
        line_start = int(
            finding.get("line_start")
        )
    except (
        TypeError,
        ValueError,
    ):
        line_start = None

    try:
        chunk_start = int(
            finding.get("chunk_line_start")
        )
    except (
        TypeError,
        ValueError,
    ):
        chunk_start = None

    if (
        line_start is not None
        and chunk_start is not None
    ):
        index = line_start - chunk_start

        if 0 <= index < len(lines):
            return {
                "found": True,
                "index": index,
                "line": lines[index],
            }

    # --------------------------------------------------------
    # Try current_code exact match
    # --------------------------------------------------------

    reported_code = str(
        finding.get("current_code")
        or ""
    ).strip()

    if reported_code:
        reported_lines = reported_code.splitlines()

        # Exact multi-line block
        if reported_lines:
            for index in range(
                0,
                len(lines) - len(reported_lines) + 1,
            ):
                block = lines[
                    index:index + len(reported_lines)
                ]

                if [
                    line.strip()
                    for line in block
                ] == [
                    line.strip()
                    for line in reported_lines
                ]:
                    return {
                        "found": True,
                        "index": index,
                        "line": lines[index],
                    }

        # Single-line fallback
        for index, line in enumerate(lines):
            if line.strip() == reported_code:
                return {
                    "found": True,
                    "index": index,
                    "line": line,
                }

    return {
        "found": False,
        "index": None,
        "line": None,
    }


# ============================================================
# VALIDATE REMOVE ACTION
# ============================================================

def _validate_remove(
    file_path: str,
    original_code: str,
    finding: Dict[str, Any],
):
    """
    Validate a REMOVE correction.

    A remove action is valid only when the reported code/line
    can actually be located in the supplied source.
    """

    if not original_code:
        return {
            "valid": False,
            "message": (
                "Remove action cannot be validated because "
                "the original code is missing."
            ),
            "language": "unknown",
            "validation_method": "remove-location",
            "corrected_code": None,
        }

    location = _find_reported_line(
        original_code,
        finding,
    )

    if not location["found"]:
        return {
            "valid": False,
            "message": (
                "Remove action could not locate the reported "
                "code inside the current code."
            ),
            "language": "unknown",
            "validation_method": "remove-location",
            "corrected_code": None,
        }

    lines = str(original_code).splitlines()

    index = location["index"]

    if index is None or not (
        0 <= index < len(lines)
    ):
        return {
            "valid": False,
            "message": "Reported removal line is invalid.",
            "language": "unknown",
            "validation_method": "remove-location",
            "corrected_code": None,
        }

    corrected_lines = (
        lines[:index]
        + lines[index + 1:]
    )

    corrected_code = "\n".join(
        corrected_lines
    )

    # If there was only one line, empty result is acceptable
    # as long as the reported line was actually found.
    if corrected_code.strip():
        syntax_validation = _validate_code(
            file_path,
            corrected_code,
        )

        if not syntax_validation["valid"]:
            return {
                "valid": False,
                "message": (
                    "The code becomes syntactically invalid "
                    "after removing the reported line. "
                    + syntax_validation["message"]
                ),
                "language": syntax_validation["language"],
                "validation_method": (
                    "remove + "
                    + syntax_validation["validation_method"]
                ),
                "corrected_code": corrected_code,
            }

    return {
        "valid": True,
        "message": (
            "The reported code was located and can be "
            "removed without creating a structural "
            "syntax error."
        ),
        "language": _validate_code(
            file_path,
            original_code,
        )["language"],
        "validation_method": "remove + syntax validation",
        "corrected_code": corrected_code,
    }


# ============================================================
# VALIDATE ADD ACTION
# ============================================================

def _validate_add(
    file_path: str,
    result: Dict[str, Any],
):
    """
    Validate an ADD correction.

    The added code must exist and be syntactically valid.
    An explicit location is also required.
    """

    replacement_code = str(
        result.get("replacement_code")
        or ""
    ).strip()

    add_location = str(
        result.get("add_location")
        or ""
    ).strip()

    if not replacement_code:
        return {
            "valid": False,
            "message": (
                "Add action requires the actual code "
                "that must be added."
            ),
            "language": "unknown",
            "validation_method": "add-code",
        }

    if not add_location:
        return {
            "valid": False,
            "message": (
                "Add action requires an exact location "
                "such as a line, function, or statement "
                "where the code should be inserted."
            ),
            "language": "unknown",
            "validation_method": "add-location",
        }

    validation = _validate_code(
        file_path,
        replacement_code,
    )

    if not validation["valid"]:
        return {
            "valid": False,
            "message": (
                "The code proposed for addition is not "
                "syntactically valid. "
                + validation["message"]
            ),
            "language": validation["language"],
            "validation_method": (
                "add + "
                + validation["validation_method"]
            ),
        }

    return {
        "valid": True,
        "message": (
            "The code proposed for addition is "
            "syntactically valid and an insertion "
            "location was provided."
        ),
        "language": validation["language"],
        "validation_method": (
            "add + "
            + validation["validation_method"]
        ),
    }


# ============================================================
# VALIDATE REPLACE ACTION
# ============================================================

def _validate_replace(
    file_path: str,
    original_code: str,
    replacement_code: str,
):
    """
    Validate a REPLACE correction.

    The replacement must:
      1. exist,
      2. differ from the original,
      3. be syntactically valid.
    """

    if not replacement_code or not replacement_code.strip():
        return {
            "valid": False,
            "message": (
                "Replace action requires actual "
                "replacement code."
            ),
            "language": "unknown",
            "validation_method": "replace-code",
        }

    replacement_code = str(
        replacement_code
    ).strip()

    if (
        original_code
        and replacement_code
        == str(original_code).strip()
    ):
        return {
            "valid": False,
            "message": (
                "Replacement code is identical to the "
                "original code, so no correction was made."
            ),
            "language": "unknown",
            "validation_method": "replace-comparison",
        }

    validation = _validate_code(
        file_path,
        replacement_code,
    )

    if not validation["valid"]:
        return {
            "valid": False,
            "message": (
                "Replacement code failed syntax "
                "validation. "
                + validation["message"]
            ),
            "language": validation["language"],
            "validation_method": (
                "replace + "
                + validation["validation_method"]
            ),
        }

    return {
        "valid": True,
        "message": (
            "Replacement code is different from the "
            "reported code and passed syntax validation."
        ),
        "language": validation["language"],
        "validation_method": (
            "replace + "
            + validation["validation_method"]
        ),
    }


# ============================================================
# MAIN VALIDATION FUNCTION
# ============================================================

def validate_replacement_code(
    file_path,
    original_code,
    replacement_code,
):
    """
    Backwards-compatible validation function.

    This function is kept because older code may call it
    directly.

    It validates replacement code only.
    """

    return _validate_replace(
        file_path,
        original_code,
        replacement_code,
    )


# ============================================================
# VALIDATE COMPLETE CORRECTION
# ============================================================

def validate_correction(
    file_path: str,
    original_code: str,
    result: Dict[str, Any],
    finding: Dict[str, Any],
):
    """
    Validate the complete correction action.

    Supported actions:

        replace
        remove
        add
    """

    if not isinstance(result, dict):
        result = {}

    if not isinstance(finding, dict):
        finding = {}

    solution_type = _normalize_solution_type(
        result,
        finding,
    )

    if solution_type == "remove":
        validation = _validate_remove(
            file_path,
            original_code,
            finding,
        )

    elif solution_type == "add":
        validation = _validate_add(
            file_path,
            result,
        )

    else:
        validation = _validate_replace(
            file_path,
            original_code,
            result.get(
                "replacement_code"
            ),
        )

    validation["solution_type"] = solution_type

    return validation


# ============================================================
# APPLY VALIDATION TO LLM RESULT
# ============================================================

def apply_validation_to_result(
    result,
    finding,
):
    """
    Add validation information to the LLM result.

    This function preserves the existing result dictionary.

    It does NOT claim that syntax validation proves the
    logical correctness of the fix.
    """

    if not isinstance(result, dict):
        result = {}

    if not isinstance(finding, dict):
        finding = {}

    # --------------------------------------------------------
    # Original code
    # --------------------------------------------------------

    original_code = (
        finding.get("current_code")
        or finding.get("code_chunk")
        or finding.get("chunk_context")
        or ""
    )

    original_code = str(
        original_code or ""
    )

    # --------------------------------------------------------
    # Normalize action
    # --------------------------------------------------------

    solution_type = _normalize_solution_type(
        result,
        finding,
    )

    # --------------------------------------------------------
    # Validate correction
    # --------------------------------------------------------

    validation = validate_correction(
        finding.get("file", ""),
        original_code,
        result,
        finding,
    )

    # --------------------------------------------------------
    # Store validation information
    # --------------------------------------------------------

    result["validation_status"] = (
        "Passed"
        if validation["valid"]
        else "Failed"
    )

    result["validation_message"] = validation[
        "message"
    ]

    result["validation_method"] = validation[
        "validation_method"
    ]

    result["validation_language"] = validation[
        "language"
    ]

    result["fix_validated"] = bool(
        validation["valid"]
    )

    # --------------------------------------------------------
    # Keep correction action explicit
    # --------------------------------------------------------

    result["solution_type"] = solution_type

    # --------------------------------------------------------
    # REMOVE action
    # --------------------------------------------------------

    if solution_type == "remove":

        location = _find_reported_line(
            original_code,
            finding,
        )

        if location["found"]:

            result["replacement_code"] = ""

            result["solution"] = (
                result.get("solution")
                or
                "Remove the reported line because "
                "it is unnecessary or causes the "
                "detected problem."
            )

            result["explanation"] = (
                result.get("explanation")
                or
                "The reported code was located in "
                "the current source and can be removed."
            )

        else:
            result["insufficient_evidence"] = True

    # --------------------------------------------------------
    # ADD action
    # --------------------------------------------------------

    elif solution_type == "add":

        if not result.get("replacement_code"):
            result["insufficient_evidence"] = True

        if not result.get("add_location"):
            result["insufficient_evidence"] = True

        if not result.get("solution"):
            result["solution"] = (
                "Add the corrected code at the specified "
                "location to handle the detected issue."
            )

    # --------------------------------------------------------
    # REPLACE action
    # --------------------------------------------------------

    else:

        if not result.get("replacement_code"):
            result["insufficient_evidence"] = True

        if not result.get("solution"):
            result["solution"] = (
                "Replace the reported code with the "
                "corrected code shown below."
            )

    # --------------------------------------------------------
    # Confidence handling
    # --------------------------------------------------------

    try:
        current_confidence = int(
            result.get(
                "confidence",
                70,
            )
        )
    except (
        TypeError,
        ValueError,
    ):
        current_confidence = 70

    current_confidence = max(
        0,
        min(
            100,
            current_confidence,
        ),
    )

    # IMPORTANT:
    #
    # Syntax validation alone does not prove that the
    # correction is logically correct.
    #
    # Therefore:
    #
    #   valid fix      -> preserve/increase only slightly
    #   invalid fix    -> confidence capped
    #
    if validation["valid"]:

        result["confidence"] = max(
            current_confidence,
            70,
        )

        if (
            current_confidence >= 70
            and not result.get(
                "insufficient_evidence",
                False,
            )
        ):
            result["confidence_level"] = (
                "High Confidence"
            )
        else:
            result["confidence_level"] = (
                "Low Confidence"
            )

    else:

        result["confidence"] = min(
            current_confidence,
            59,
        )

        result["confidence_level"] = (
            "Low Confidence"
        )

        result["insufficient_evidence"] = True

    # --------------------------------------------------------
    # Confidence status for frontend
    # --------------------------------------------------------

    if result.get("confidence_level") == "High Confidence":
        result["confidence_status"] = (
            "Potential Bug"
        )
    else:
        result["confidence_status"] = (
            "Uncertain Finding"
        )

    return result