"""
Scans every file in the downloaded repository.

All files are discovered, including files without a known source-code
extension. Binary files are detected and safely skipped from text/LLM
analysis because they are not source text.
"""

import os


def find_source_files(root_path: str):
    """Return absolute paths for every file in the repository."""
    found = []

    for dirpath, dirnames, filenames in os.walk(root_path):
        # Do not remove folders. We want every repository file.
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            found.append(full_path)

    return found


def is_binary_file(path: str) -> bool:
    """
    Return True when the file appears to contain binary data.

    A null byte is a strong indicator that the file is binary.
    Small binary files are also checked using a simple byte heuristic.
    """
    try:
        with open(path, "rb") as f:
            chunk = f.read(8192)

        if not chunk:
            return False

        # Most normal source/text files do not contain null bytes.
        if b"\x00" in chunk:
            return True

        # Check for a high percentage of non-text bytes.
        suspicious = 0

        for byte in chunk:
            if byte < 9 or (13 < byte < 32):
                suspicious += 1

        return (suspicious / len(chunk)) > 0.30

    except OSError:
        return True


def read_file_safely(path: str) -> str:
    """
    Read a text file safely.

    Binary files return an empty string so they are never sent to the LLM.
    """
    if is_binary_file(path):
        return ""

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    except OSError:
        return ""
