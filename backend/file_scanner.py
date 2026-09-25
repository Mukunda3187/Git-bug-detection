
"""
Source file scanner for the Git Bug Detection project.

Finds supported source files, skips unnecessary directories and binary
files, and avoids reading oversized files into memory.
"""

import os


# Supported programming-language extensions.
SOURCE_EXTENSIONS = {
    ".py", ".pyi",
    ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    ".java",
    ".c", ".h", ".cpp", ".hpp", ".cc", ".hh",
    ".cs", ".go", ".php", ".rb", ".rs",
    ".kt", ".kts", ".swift", ".scala", ".sql",
}

# Directories that should not be scanned.
SKIP_DIRS = {
    ".git",
    ".github",
    ".idea",
    ".vscode",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "dist",
    "build",
    "coverage",
    ".next",
    ".nuxt",
    "vendor",
    "target",
    "site-packages",
    "bower_components",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}

# Files that are usually generated or not useful for source analysis.
SKIP_SUFFIXES = (
    ".min.js",
    ".min.css",
    ".map",
    ".lock",
    ".svg",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".pdf",
    ".zip",
    ".gz",
    ".tar",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
)

MAX_FILE_SIZE = 300_000  # 300 KB


def _is_skipped_directory(path_parts):
    """Return True if any directory component should be skipped."""
    return any(
        part.lower() in SKIP_DIRS
        for part in path_parts
    )


def _is_supported_source_file(filename):
    """Check whether a filename has a supported source extension."""
    lower_name = filename.lower()

    if lower_name.endswith(SKIP_SUFFIXES):
        return False

    _, extension = os.path.splitext(lower_name)

    return extension in SOURCE_EXTENSIONS


def find_source_files(root_path: str):
    """
    Return source file paths under root_path.

    Skips generated directories, unsupported file types, and oversized files.
    """
    found = []

    if not root_path or not os.path.isdir(root_path):
        return found

    for dirpath, dirnames, filenames in os.walk(root_path):

        # Prune excluded directories before os.walk enters them.
        dirnames[:] = sorted(
            directory
            for directory in dirnames
            if directory.lower() not in SKIP_DIRS
        )

        for filename in sorted(filenames):
            if not _is_supported_source_file(filename):
                continue

            full_path = os.path.join(dirpath, filename)

            try:
                file_size = os.path.getsize(full_path)
            except OSError:
                continue

            if file_size <= 0 or file_size > MAX_FILE_SIZE:
                continue

            found.append(full_path)

    print(f"[scanner] Found {len(found)} supported source files.")

    return found


def is_binary_file(path: str) -> bool:
    """Return True if a file appears to contain binary data."""
    try:
        with open(path, "rb") as handle:
            chunk = handle.read(8192)

        if not chunk:
            return False

        # Null bytes usually indicate binary content.
        if b"\x00" in chunk:
            return True

        suspicious = sum(
            1
            for byte in chunk
            if byte < 9 or 13 < byte < 32
        )

        return (suspicious / len(chunk)) > 0.30

    except OSError:
        return True


def read_file_safely(path: str) -> str:
    """
    Read a source file as UTF-8.

    Returns an empty string for binary, oversized, or unreadable files.
    """
    if not path:
        return ""

    try:
        if os.path.getsize(path) > MAX_FILE_SIZE:
            return ""
    except OSError:
        return ""

    if is_binary_file(path):
        return ""

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read()

    except OSError as exc:
        print(f"[scanner] Could not read {path}: {exc}")
        return ""
