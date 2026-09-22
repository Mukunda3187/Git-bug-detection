"""
Walks the downloaded repository and returns a list of source files worth
analyzing. Keeps ignored folders in one config list so it's easy to tweak.

Only excludes things that are NOT the developer's own code: version
control internals, third-party dependencies, and generated build output.
Scanning those would waste time finding "bugs" in code the developer
didn't write and can't meaningfully fix (it just gets overwritten on the
next install/build), while genuinely missing real problems in the
developer's own files. Everything else the developer actually committed -
including things like database migrations - gets scanned, since it's
real, versioned code that can have real bugs.
"""
import os


# A generous ceiling mainly to protect against something that's clearly not
# meant to be read as source (a huge minified bundle, a data file with a
# misleading extension) rather than to skip legitimately large source files.


def find_source_files(root_path: str):
    """Returns a list of absolute paths to source files worth scanning."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames[:] = dirnames
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            found.append(full_path)
    return found


def read_file_safely(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except OSError:
        return ""
