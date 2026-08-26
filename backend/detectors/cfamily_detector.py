"""
Detector for Java, C, C++, C#, Go, and PHP.

No compiler for any of these (javac, gcc/g++, the Go toolchain, php -l,
a C# compiler) is installed in this Python-only backend, so - same
approach as js_detector.py - this checks the one thing a pure-Python
pass can check reliably across all of them: do the brackets, strings,
and comments in this file actually balance. That's the class of error
that stops a build cold, and it's the same engine used for JS/TS
(detectors/syntax_balance.py), just pointed at each language's own
comment/string rules.

Known, honest limitations (documented rather than silently wrong):
- PHP's rare backtick shell-exec operator (`` `cmd` ``) and C#'s raw
  string literals (\"\"\"...\"\"\") aren't modeled - both are uncommon
  enough in real code that they're a fine trade-off for now.
- This only catches structural/bracket-level problems, not
  language-specific compile errors (type errors, missing semicolons,
  undeclared variables, etc.). Those need each language's real
  compiler - see the README section on the phased plan for adding
  compiler-backed checks next.
"""
from .syntax_balance import check_bracket_balance, GO_CONFIG, C_FAMILY_CONFIG

CONFIG_BY_EXTENSION = {
    ".java": C_FAMILY_CONFIG,
    ".c": C_FAMILY_CONFIG,
    ".cpp": C_FAMILY_CONFIG,
    ".cs": C_FAMILY_CONFIG,
    ".php": C_FAMILY_CONFIG,
    ".go": GO_CONFIG,
}


def detect(file_path: str, source: str):
    ext = file_path[file_path.rfind("."):] if "." in file_path else ""
    config = CONFIG_BY_EXTENSION.get(ext, C_FAMILY_CONFIG)

    structural = check_bracket_balance(source, config)
    if not structural:
        return []

    finding = dict(structural[0])
    finding["file"] = file_path
    finding["function"] = None
    return [finding]
