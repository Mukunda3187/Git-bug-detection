"""
Source-file scanner for Git Bug Detection.

This module discovers ALL supported source-code files inside a downloaded
GitHub repository.

There is intentionally NO fixed file-count limit.

There is intentionally NO artificial source-file-size limit here.

Large files are still handled safely:
- binary files are skipped
- unreadable files are skipped
- generated/dependency directories are skipped
- files are read only when the main scan actually needs their contents
"""

import os


# ============================================================
# SUPPORTED SOURCE EXTENSIONS
# ============================================================

SUPPORTED_EXTENSIONS = {
    # Python
    ".py",
    ".pyi",

    # JavaScript / TypeScript
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".mjs",
    ".cjs",

    # Java
    ".java",

    # C / C++
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".cc",
    ".hh",
    ".cxx",
    ".hxx",

    # C#
    ".cs",

    # Go
    ".go",

    # PHP
    ".php",

    # Ruby
    ".rb",

    # Rust
    ".rs",

    # Kotlin
    ".kt",
    ".kts",

    # Swift
    ".swift",

    # Scala
    ".scala",

    # SQL
    ".sql",

    # Shell
    ".sh",
    ".bash",

    # Dart
    ".dart",

    # R
    ".r",

    # Lua
    ".lua",

    # Perl
    ".pl",
    ".pm",

    # Objective-C
    ".m",
    ".mm",

    # Groovy
    ".groovy",

    # Elixir
    ".ex",
    ".exs",

    # Haskell
    ".hs",

    # Julia
    ".jl",
}


# ============================================================
# DIRECTORIES THAT SHOULD NOT BE SCANNED
# ============================================================

IGNORED_FOLDERS = {
    ".git",
    ".github",
    ".idea",
    ".vscode",

    "__pycache__",

    "node_modules",

    "venv",
    ".venv",
    "env",

    "dist",
    "build",
    "out",

    "target",

    "vendor",

    "coverage",

    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",

    ".gradle",
    ".terraform",

    ".next",
    ".nuxt",

    ".angular",

    "bower_components",

    "site-packages",

    "bin",
    "obj",
}


# ============================================================
# GENERATED / MINIFIED FILE NAMES
# ============================================================

IGNORED_FILE_SUFFIXES = (
    ".min.js",
    ".min.ts",
    ".map",
)


# ============================================================
# PATH HELPERS
# ============================================================

def _is_ignored_directory(name: str) -> bool:
    """
    Check whether a directory should be excluded.
    """

    if not name:
        return True

    if name in IGNORED_FOLDERS:
        return True

    # Hidden build/cache directories that are not useful
    # for repository source analysis.
    if name.startswith(".") and name not in {
        ".config",
    }:
        return True

    return False


def _is_supported_file(filename: str) -> bool:
    """
    Return True when the filename represents a supported source file.
    """

    if not filename:
        return False

    lower_name = filename.lower()

    for suffix in IGNORED_FILE_SUFFIXES:

        if lower_name.endswith(suffix):
            return False

    _, extension = os.path.splitext(
        lower_name
    )

    return extension in SUPPORTED_EXTENSIONS


# ============================================================
# BINARY FILE DETECTION
# ============================================================

def _is_binary_file(path: str) -> bool:
    """
    Check whether a file appears to be binary.

    Only a small initial portion is read, so this does not load
    the entire file into memory.
    """

    try:

        with open(
            path,
            "rb",
        ) as handle:

            sample = handle.read(
                8192
            )

    except OSError:

        return True

    if not sample:
        return False

    # A NULL byte is a strong indication of binary content.
    if b"\x00" in sample:
        return True

    # UTF-8 validation.
    try:

        sample.decode(
            "utf-8"
        )

    except UnicodeDecodeError:

        return True

    return False


# ============================================================
# FIND SOURCE FILES
# ============================================================

def find_source_files(root_path: str):
    """
    Find ALL supported source files in the repository.

    No fixed file-count limit is applied.

    Returns:
        List of absolute file paths.
    """

    found = []

    if not root_path:
        return found

    if not os.path.isdir(root_path):
        return found

    for dirpath, dirnames, filenames in os.walk(
        root_path,
        topdown=True,
    ):

        # ----------------------------------------------------
        # PRUNE IGNORED DIRECTORIES
        # ----------------------------------------------------

        dirnames[:] = [
            dirname

            for dirname in dirnames

            if not _is_ignored_directory(
                dirname
            )
        ]

        # ----------------------------------------------------
        # PROCESS FILES
        # ----------------------------------------------------

        for filename in filenames:

            if not _is_supported_file(
                filename
            ):
                continue

            full_path = os.path.join(
                dirpath,
                filename,
            )

            # ------------------------------------------------
            # BASIC FILE VALIDATION
            # ------------------------------------------------

            try:

                if not os.path.isfile(
                    full_path
                ):
                    continue

                # Ignore broken/special files.
                if os.path.islink(
                    full_path
                ):
                    continue

            except OSError:
                continue

            # ------------------------------------------------
            # BINARY CHECK
            # ------------------------------------------------

            if _is_binary_file(
                full_path
            ):
                continue

            found.append(
                full_path
            )

    # Stable ordering makes:
    # - bug numbering predictable
    # - scan results reproducible
    # - second scans easier to compare
    found.sort(
        key=lambda path: os.path.relpath(
            path,
            root_path,
        ).lower()
    )

    print(
        "[scanner] Source files found: "
        f"{len(found)}"
    )

    return found


# ============================================================
# SAFE FILE READING
# ============================================================

def read_file_safely(path: str) -> str:
    """
    Read a source file as UTF-8.

    The file is read only when requested by the scan worker.

    Invalid UTF-8 bytes are ignored rather than crashing the
    entire repository scan.
    """

    if not path:
        return ""

    try:

        if not os.path.isfile(
            path
        ):
            return ""

    except OSError:
        return ""

    try:

        with open(
            path,
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as handle:

            return handle.read()

    except (
        OSError,
        UnicodeError,
    ):

        return ""


# ============================================================
# RELATIVE PATH HELPER
# ============================================================

def relative_source_path(
    root_path: str,
    file_path: str,
) -> str:
    """
    Return a normalized repository-relative path.
    """

    try:

        return os.path.relpath(
            file_path,
            root_path,
        ).replace(
            os.sep,
            "/",
        )

    except (
        ValueError,
        TypeError,
    ):

        return file_path
