"""
Detector for Java, C, C++, C#, Go, PHP, Rust, Kotlin, Swift, and Ruby.

Uses pure-Python structural checks and safe pattern-based heuristics.
It does not execute or compile repository code.
"""

import re

from .syntax_balance import (
    check_bracket_balance,
    mask_non_code,
    GO_CONFIG,
    C_FAMILY_CONFIG,
)

# ---------------------------------------------------------
# Language configurations
# ---------------------------------------------------------

CONFIG_BY_EXTENSION = {
    ".java": C_FAMILY_CONFIG,
    ".c": C_FAMILY_CONFIG,
    ".cpp": C_FAMILY_CONFIG,
    ".cs": C_FAMILY_CONFIG,
    ".php": C_FAMILY_CONFIG,
    ".go": GO_CONFIG,

    # These languages use the same structural bracket/string
    # checking available in the current detector.
    ".rs": C_FAMILY_CONFIG,   # Rust
    ".kt": C_FAMILY_CONFIG,   # Kotlin
    ".kts": C_FAMILY_CONFIG,  # Kotlin Script
    ".swift": C_FAMILY_CONFIG,
    ".rb": C_FAMILY_CONFIG,   # Ruby
}


# ---------------------------------------------------------
# Unreachable-code checking
# ---------------------------------------------------------

UNREACHABLE_CHECK_EXTENSIONS = {
    ".java",
    ".c",
    ".cpp",
    ".cs",
    ".php",
    ".rs",
    ".kt",
    ".kts",
    ".swift",
}


# ---------------------------------------------------------
# Empty catch block
# ---------------------------------------------------------

EMPTY_CATCH_EXTENSIONS = {
    ".java",
    ".cs",
    ".cpp",
    ".php",
    ".kt",
    ".kts",
}


EMPTY_CATCH_RE = re.compile(
    r"catch\s*(\([^)]*\))?\s*\{\s*\}"
)


# ---------------------------------------------------------
# Debug-print patterns
# ---------------------------------------------------------

DEBUG_PRINT_PATTERNS = {

    # Java
    ".java": re.compile(
        r"\bSystem\s*\.\s*(out|err)\s*\.\s*(println|print)\s*\("
    ),

    # C#
    ".cs": re.compile(
        r"\bConsole\s*\.\s*(WriteLine|Write)\s*\("
    ),

    # PHP
    ".php": re.compile(
        r"\b(var_dump|print_r)\s*\("
    ),

    # Kotlin
    ".kt": re.compile(
        r"\bprintln\s*\("
    ),

    ".kts": re.compile(
        r"\bprintln\s*\("
    ),

    # Swift
    ".swift": re.compile(
        r"\bprint\s*\("
    ),

    # Ruby
    ".rb": re.compile(
        r"\b(puts|p)\s*\(?"
    ),
}


# ---------------------------------------------------------
# Rust debug patterns
# ---------------------------------------------------------

RUST_DEBUG_PRINT_RE = re.compile(
    r"\b(?:println!|print!|dbg!)\s*\("
)


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def _line_of(source: str, index: int) -> int:
    return source.count("\n", 0, index) + 1


def _line_text(lines, ln):
    return (
        lines[ln - 1].strip()
        if 0 < ln <= len(lines)
        else ""
    )


# ---------------------------------------------------------
# Pattern-based findings
# ---------------------------------------------------------

def _pattern_findings(
    source: str,
    file_path: str,
    ext: str,
    config
):
    masked = mask_non_code(source, config)

    lines = source.splitlines()

    findings = []

    # -----------------------------------------------------
    # Empty catch
    # -----------------------------------------------------

    if ext in EMPTY_CATCH_EXTENSIONS:

        for m in EMPTY_CATCH_RE.finditer(masked):

            ln = _line_of(
                source,
                m.start()
            )

            findings.append({
                "file": file_path,
                "function": None,
                "line_start": ln,
                "line_end": ln,
                "rule": "empty_catch_block",
                "error": "Empty catch block",
                "bug_type": "Logic Error",
                "current_code": _line_text(
                    lines,
                    ln
                ),
                "cause": (
                    "This catch block does nothing, so if "
                    "an error happens here it is silently "
                    "swallowed with no record of it anywhere."
                ),
            })

    # -----------------------------------------------------
    # Debug print
    # -----------------------------------------------------

    debug_pattern = DEBUG_PRINT_PATTERNS.get(ext)

    if debug_pattern:

        for m in debug_pattern.finditer(masked):

            ln = _line_of(
                source,
                m.start()
            )

            findings.append({
                "file": file_path,
                "function": None,
                "line_start": ln,
                "line_end": ln,
                "rule": "leftover_debug_print",
                "error": "Possible debug print statement",
                "bug_type": "Unnecessary Code",
                "current_code": _line_text(
                    lines,
                    ln
                ),
                "cause": (
                    "This looks like it might be a debugging "
                    "print statement left in the code. It could "
                    "also be intentional output - review before "
                    "removing."
                ),
            })

    # -----------------------------------------------------
    # Rust debug output
    # -----------------------------------------------------

    if ext == ".rs":

        for m in RUST_DEBUG_PRINT_RE.finditer(masked):

            ln = _line_of(
                source,
                m.start()
            )

            findings.append({
                "file": file_path,
                "function": None,
                "line_start": ln,
                "line_end": ln,
                "rule": "leftover_debug_print",
                "error": "Possible debug print statement",
                "bug_type": "Unnecessary Code",
                "current_code": _line_text(
                    lines,
                    ln
                ),
                "cause": (
                    "This Rust print or debug macro may have "
                    "been left behind during debugging. Review "
                    "whether it is intentional before removing it."
                ),
            })

    return findings


# ---------------------------------------------------------
# Main detector
# ---------------------------------------------------------

def detect(
    file_path: str,
    source: str
):
    ext = (
        file_path[file_path.rfind("."):]
        if "." in file_path
        else ""
    )

    ext = ext.lower()

    config = CONFIG_BY_EXTENSION.get(
        ext,
        C_FAMILY_CONFIG
    )

    # -----------------------------------------------------
    # Structural syntax checking
    # -----------------------------------------------------

    structural, unreachable = check_bracket_balance(
        source,
        config
    )

    if structural:

        finding = dict(
            structural[0]
        )

        finding["file"] = file_path
        finding["function"] = None

        return [finding]

    findings = []

    # -----------------------------------------------------
    # Unreachable code
    # -----------------------------------------------------

    if ext in UNREACHABLE_CHECK_EXTENSIONS:

        for f in unreachable:

            f = dict(f)

            f["file"] = file_path
            f["function"] = None

            findings.append(f)

    # -----------------------------------------------------
    # Pattern checks
    # -----------------------------------------------------

    findings.extend(
        _pattern_findings(
            source,
            file_path,
            ext,
            config
        )
    )

    return findings
