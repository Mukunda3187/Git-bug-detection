"""
Detector for Java, C, C++, C#, Go, and PHP.

No compiler for any of these (javac, gcc/g++, the Go toolchain, php -l,
a C# compiler) is installed in this Python-only backend, so this checks
everything a pure-Python pass can check reliably:

1. Structural syntax (brackets, strings, comments balance) - the class
   of error that stops a build cold - for all six languages.

2. Unreachable code (statements after an unconditional return/throw in
   the same block) - for Java, C, C++, C#, and PHP only, NOT Go. Go is
   excluded on purpose: idiomatic Go typically omits semicolons entirely
   (gofmt strips them), and this checker's unreachable-code logic relies
   on an explicit ';' to know where a statement ends.

3. Pattern-based heuristic checks, run on a version of the source with
   comment/string content masked out, for the languages where each
   pattern is actually a meaningful smell rather than idiomatic style:
   - empty_catch_block: Java, C#, C++, PHP (all have try/catch syntax;
     C and Go don't have exceptions, so this doesn't apply to them)
   - leftover_debug_print: System.out.println/System.err.println (Java),
     Console.WriteLine/Write (C#), var_dump/print_r (PHP). Deliberately
     NOT applied to C/C++ printf/cout or Go fmt.Println - those are the
     normal, everyday way those languages produce real program output,
     with no separate "console.log-style" debug convention the way
     JS/Java/C# have, so flagging every one would mostly be noise.

   These pattern checks are judgment calls, not certainties - see
   llm_client.FALLBACK_CONFIDENCE_BY_RULE for the honest confidence
   score attached to each one, distinct from the 100%-certain structural
   and unreachable-code findings above.

Known, honest limitations (documented rather than silently wrong):
- PHP's rare backtick shell-exec operator (`` `cmd` ``) and C#'s raw
  string literals (\"\"\"...\"\"\") aren't modeled - both are uncommon
  enough in real code that they're a fine trade-off for now.
- This doesn't catch language-specific compile errors (type errors,
  undeclared variables, etc.) - those need each language's real
  compiler, which isn't available in this environment.
"""
import re

from .syntax_balance import check_bracket_balance, mask_non_code, GO_CONFIG, C_FAMILY_CONFIG

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

EMPTY_CATCH_EXTENSIONS = {".java", ".cs", ".cpp", ".php"}
EMPTY_CATCH_RE = re.compile(r"catch\s*(\([^)]*\))?\s*\{\s*\}")

DEBUG_PRINT_PATTERNS = {
    ".java": re.compile(r"\bSystem\s*\.\s*(out|err)\s*\.\s*(println|print)\s*\("),
    ".cs": re.compile(r"\bConsole\s*\.\s*(WriteLine|Write)\s*\("),
    ".php": re.compile(r"\b(var_dump|print_r)\s*\("),
}


def _line_of(source: str, index: int) -> int:
    return source.count("\n", 0, index) + 1


def _line_text(lines, ln):
    return lines[ln - 1].strip() if 0 < ln <= len(lines) else ""


def _pattern_findings(source: str, file_path: str, ext: str, config):
    masked = mask_non_code(source, config)
    lines = source.splitlines()
    findings = []

    if ext in EMPTY_CATCH_EXTENSIONS:
        for m in EMPTY_CATCH_RE.finditer(masked):
            ln = _line_of(source, m.start())
            findings.append({
                "file": file_path, "function": None,
                "line_start": ln, "line_end": ln,
                "rule": "empty_catch_block",
                "error": "Empty catch block",
                "bug_type": "Logic Error",
                "current_code": _line_text(lines, ln),
                "cause": "This catch block does nothing, so if an error happens here it is silently "
                         "swallowed with no record of it anywhere.",
            })

    debug_pattern = DEBUG_PRINT_PATTERNS.get(ext)
    if debug_pattern:
        for m in debug_pattern.finditer(masked):
            ln = _line_of(source, m.start())
            findings.append({
                "file": file_path, "function": None,
                "line_start": ln, "line_end": ln,
                "rule": "leftover_debug_print",
                "error": "Possible debug print statement",
                "bug_type": "Unnecessary Code",
                "current_code": _line_text(lines, ln),
                "cause": "This looks like it might be a debugging print statement left in the code. "
                         "It could also be intentional output - review before removing.",
            })

    return findings


def detect(file_path: str, source: str):
    ext = file_path[file_path.rfind("."):] if "." in file_path else ""
    config = CONFIG_BY_EXTENSION.get(ext, C_FAMILY_CONFIG)

    structural, unreachable = check_bracket_balance(source, config)

    if structural:
        finding = dict(structural[0])
        finding["file"] = file_path
        finding["function"] = None
        return [finding]

    findings = []

    if ext in UNREACHABLE_CHECK_EXTENSIONS:
        for f in unreachable:
            f = dict(f)
            f["file"] = file_path
            f["function"] = None
            findings.append(f)

    findings.extend(_pattern_findings(source, file_path, ext, config))
    return findings
