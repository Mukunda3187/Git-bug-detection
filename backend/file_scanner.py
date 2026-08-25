"""
Walks the downloaded repository and returns a list of source files worth
analyzing. Keeps ignored folders in one config list so it's easy to tweak.
"""
import os

SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".c", ".cpp", ".cs", ".go", ".php"
}

IGNORED_FOLDERS = {
    ".git", "node_modules", "__pycache__", "venv", ".venv", "env",
    "dist", "build", "target", "vendor", ".idea", ".vscode",
    "coverage", ".pytest_cache", "migrations",
}

MAX_FILE_SIZE_BYTES = 500_000  # skip anything unusually large (likely generated/minified)


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
