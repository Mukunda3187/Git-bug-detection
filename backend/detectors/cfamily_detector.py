"""
Detector for Java, C, C++, C#, Go, and PHP.

No compiler for any of these (javac, gcc/g++, the Go toolchain, php -l,
a C# compiler) is installed in this Python-only backend, so - same
approach as js_detector.py - this checks two things a pure-Python pass
can check reliably:

1. Structural syntax (brackets, strings, comments balance) - the class
   of error that stops a build cold - for all six languages.

2. Unreachable code (statements after an unconditional return/throw in
   the same block) - for Java, C, C++, C#, and PHP only, NOT Go. Go is
   excluded on purpose: idiomatic Go typically omits semicolons entirely
   (gofmt strips them), and this checker's unreachable-code logic relies
   on an explicit ';' to know where a statement ends. Running it on Go
   would either miss everything (harmless but pointless) or need a
   different, newline-aware statement-boundary rule this project hasn't
   built yet - see syntax_balance.find_unreachable_statements for the
   full reasoning on what is and isn't flagged.

Known, honest limitations (documented rather than silently wrong):
- PHP's rare backtick shell-exec operator (`` `cmd` ``) and C#'s raw
  string literals (\"\"\"...\"\"\") aren't modeled - both are uncommon
  enough in real code that they're a fine trade-off for now.
- This only catches structural/bracket-level problems and unreachable
  code, not language-specific compile errors (type errors, missing
  semicolons, undeclared variables, etc.). Those need each language's
  real compiler - see the README section on the phased plan for adding
  compiler-backed checks next.
"""
from .syntax_balance import check_bracket_balance, find_unreachable_statements, GO_CONFIG, C_FAMILY_CONFIG

CONFIG_BY_EXTENSION = {
    ".java": C_FAMILY_CONFIG,
    ".c": C_FAMILY_CONFIG,
    ".cpp": C_FAMILY_CONFIG,
    ".cs": C_FAMILY_CONFIG,
    ".php": C_FAMILY_CONFIG,
    ".go": GO_CONFIG,
}

# Go is deliberately excluded from unreachable-code checking - see the
# module docstring above for why.
UNREACHABLE_CHECK_EXTENSIONS = {".java", ".c", ".cpp", ".cs", ".php"}


def detect(file_path: str, source: str):
    ext = file_path[file_path.rfind("."):] if "." in file_path else ""
    config = CONFIG_BY_EXTENSION.get(ext, C_FAMILY_CONFIG)

    structural = check_bracket_balance(source, config)
    if structural:
        finding = dict(structural[0])
        finding["file"] = file_path
        finding["function"] = None
        return [finding]

    if ext not in UNREACHABLE_CHECK_EXTENSIONS:
        return []

    findings = []
    for f in find_unreachable_statements(source, config):
        f = dict(f)
        f["file"] = file_path
        f["function"] = None
        findings.append(f)
    return findings
