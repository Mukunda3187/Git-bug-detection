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

SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".c", ".cpp", ".cs", ".go", ".php"
}

IGNORED_FOLDERS = {
    ".git",             # version control internals, not source code
    "node_modules",     # third-party JS dependencies
    "vendor",           # third-party dependencies (Go, PHP, etc.)
    "__pycache__",      # compiled Python bytecode cache
    "venv", ".venv", "env",  # Python virtual environments (not the developer's code)
    "dist", "build", "target",  # generated build output
    ".idea", ".vscode",  # editor/IDE settings, not source
    "coverage", ".pytest_cache",  # generated test-tooling output
}

# A generous ceiling mainly to protect against something that's clearly not
# meant to be read as source (a huge minified bundle, a data file with a
# misleading extension) rather than to skip legitimately large source files.
MAX_FILE_SIZE_BYTES = 5_000_000


def find_source_files(root_path: str):
    """Returns a list of absolute paths to source files worth scanning."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames[:] = [d for d in dirnames if d not in IGNORED_FOLDERS and not d.startswith(".")]
        for filename in filenames:
            ext = os.path.splitext(filename)[1]
            if ext not in SUPPORTED_EXTENSIONS:
                continue
            full_path = os.path.join(dirpath, filename)
            try:
                if os.path.getsize(full_path) > MAX_FILE_SIZE_BYTES:
                    continue
            except OSError:
                continue
            found.append(full_path)
    return found


def read_file_safely(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except OSError:
        return ""
