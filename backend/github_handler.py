"""
GitHub repository handling for Git Bug Detection.

This module:

1. Validates a GitHub repository.
2. Retrieves the repository file tree.
3. Selects ALL supported source-code files.
4. Downloads source files to a temporary directory.
5. Streams files to disk instead of loading the entire file into RAM.
6. Keeps GitHub historical Issues and Pull Requests retrieval for RAG.

IMPORTANT
---------
There is intentionally NO fixed maximum number of source files.

The purpose of this version is to allow the scanner to process all
supported source files in the repository.

Large repositories are handled by the background scan architecture
implemented in main.py.
"""

import os
import re
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote

import requests

from models import RepoStatus


# ============================================================
# GITHUB CONFIGURATION
# ============================================================

GITHUB_API = "https://api.github.com"

GITHUB_TIMEOUT = 30
RAW_TIMEOUT = 60

DOWNLOAD_WORKERS = 6

MAX_ISSUES = 30
MAX_PULL_REQUESTS = 30


# ============================================================
# SUPPORTED SOURCE EXTENSIONS
# ============================================================

SOURCE_EXTENSIONS = {
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
# DIRECTORIES THAT SHOULD NEVER BE SCANNED
# ============================================================

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

    ".tox",

    ".gradle",

    ".terraform",

    ".serverless",

    ".angular",

    "out",

    "bin",

    "obj",
}


# ============================================================
# TEMP DIRECTORY TRACKING
# ============================================================

_TEMP_ROOTS = {}


# ============================================================
# GITHUB HEADERS
# ============================================================

def _github_headers():
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "Git-Bug-Detection",
    }

    token = os.getenv("GITHUB_TOKEN")

    if token and token.strip():
        headers["Authorization"] = (
            f"Bearer {token.strip()}"
        )

    return headers


# ============================================================
# URL PARSING
# ============================================================

def parse_github_url(repo_url: str):
    if not repo_url:
        raise ValueError(
            "GitHub repository URL is required."
        )

    repo_url = repo_url.strip()

    match = re.match(
        r"^https?://(?:www\.)?github\.com/"
        r"([^/]+)/([^/#?]+?)(?:\.git)?/?$",
        repo_url,
        re.IGNORECASE,
    )

    if not match:
        raise ValueError(
            "Invalid GitHub repository URL."
        )

    owner = match.group(1)
    repo = match.group(2)

    return owner, repo


# ============================================================
# REPOSITORY VALIDATION
# ============================================================

def validate_repo(repo_url: str):
    owner, repo = parse_github_url(repo_url)

    url = (
        f"{GITHUB_API}/repos/"
        f"{owner}/{repo}"
    )

    try:
        response = requests.get(
            url,
            headers=_github_headers(),
            timeout=GITHUB_TIMEOUT,
        )

    except requests.RequestException as exc:

        raise RuntimeError(
            f"Unable to contact GitHub: {exc}"
        ) from exc

    if response.status_code == 404:
        raise ValueError(
            "GitHub repository not found."
        )

    if response.status_code == 403:
        raise RuntimeError(
            "GitHub API rate limit reached or "
            "access was denied."
        )

    if not response.ok:

        raise RuntimeError(
            "GitHub repository validation failed: "
            f"HTTP {response.status_code}"
        )

    try:
        data = response.json()

    except ValueError as exc:

        raise RuntimeError(
            "GitHub returned an invalid response."
        ) from exc

    return {
        "owner": owner,

        "repo": repo,

        "full_name": data.get(
            "full_name",
            f"{owner}/{repo}",
        ),

        "default_branch": data.get(
            "default_branch",
            "main",
        ),

        "description": data.get(
            "description"
        ),

        "html_url": data.get(
            "html_url",
            repo_url,
        ),
    }


# ============================================================
# PATH FILTERING
# ============================================================

def _should_skip_path(path: str) -> bool:

    normalized = (
        path
        .replace("\\", "/")
        .strip("/")
    )

    if not normalized:
        return True

    parts = normalized.split("/")

    # Check directory names.
    for part in parts[:-1]:

        if part.lower() in SKIP_DIRS:
            return True

    filename = parts[-1].lower()

    # Generated/minified files.
    if filename.endswith(
        (
            ".min.js",
            ".min.ts",
            ".min.css",
            ".map",
        )
    ):
        return True

    # Common generated source files.
    if filename.startswith(
        (
            "bundle.",
            "vendor.",
        )
    ):
        return True

    return False


def _is_source_path(path: str) -> bool:

    if _should_skip_path(path):
        return False

    _, extension = os.path.splitext(path)

    return (
        extension.lower()
        in SOURCE_EXTENSIONS
    )


# ============================================================
# GITHUB TREE
# ============================================================

def _fetch_repository_tree(
    owner: str,
    repo: str,
    branch: str,
):
    """
    Retrieve the complete recursive Git tree available through
    GitHub's Git Trees API.

    GitHub may mark very large trees as truncated. In that case
    we preserve all entries returned by GitHub and report the
    condition in the logs instead of silently pretending that
    every repository entry was returned.
    """

    branch_encoded = quote(
        branch,
        safe="",
    )

    url = (
        f"{GITHUB_API}/repos/"
        f"{owner}/{repo}/git/trees/"
        f"{branch_encoded}"
        f"?recursive=1"
    )

    try:
        response = requests.get(
            url,
            headers=_github_headers(),
            timeout=GITHUB_TIMEOUT,
        )

    except requests.RequestException as exc:

        raise RuntimeError(
            "Unable to retrieve repository "
            f"file list: {exc}"
        ) from exc

    if response.status_code == 404:

        raise RuntimeError(
            "Repository branch could not be found."
        )

    if response.status_code == 403:

        raise RuntimeError(
            "GitHub API rate limit reached while "
            "retrieving the repository tree."
        )

    if not response.ok:

        raise RuntimeError(
            "Unable to retrieve repository "
            f"file list: HTTP {response.status_code}"
        )

    try:
        data = response.json()

    except ValueError as exc:

        raise RuntimeError(
            "GitHub returned an invalid tree response."
        ) from exc

    if data.get("truncated"):

        print(
            "[github] WARNING: GitHub returned a "
            "truncated recursive tree."
        )

        print(
            "[github] The returned tree does not "
            "represent every repository path."
        )

    return data.get("tree", [])


# ============================================================
# SELECT ALL SUPPORTED SOURCE FILES
# ============================================================

def _candidate_files(tree):

    candidates = []

    for item in tree:

        # Only actual files.
        if item.get("type") != "blob":
            continue

        path = item.get(
            "path",
            "",
        )

        if not path:
            continue

        if not _is_source_path(path):
            continue

        size = item.get("size")

        # Some GitHub tree responses may not include size.
        if isinstance(size, int):

            if size < 0:
                continue

        else:
            size = 0

        candidates.append(
            {
                "path": path,
                "size": size,
            }
        )

    # --------------------------------------------------------
    # NO FILE COUNT LIMIT
    # --------------------------------------------------------

    total_bytes = sum(
        item["size"]
        for item in candidates
    )

    print(
        "[github] Source files selected: "
        f"{len(candidates)}"
    )

    print(
        "[github] Estimated source size: "
        f"{total_bytes / 1_000_000:.2f} MB"
    )

    return candidates


# ============================================================
# STREAM ONE FILE TO DISK
# ============================================================

def _download_one_file(
    owner,
    repo,
    branch,
    item,
    root_path,
):
    path = item["path"]

    encoded_path = quote(
        path,
        safe="/",
    )

    encoded_branch = quote(
        branch,
        safe="",
    )

    url = (
        "https://raw.githubusercontent.com/"
        f"{owner}/{repo}/"
        f"{encoded_branch}/"
        f"{encoded_path}"
    )

    destination = os.path.join(
        root_path,
        path,
    )

    os.makedirs(
        os.path.dirname(destination),
        exist_ok=True,
    )

    temporary_destination = (
        destination + ".download"
    )

    try:

        # ----------------------------------------------------
        # STREAM RESPONSE
        # ----------------------------------------------------

        response = requests.get(
            url,
            timeout=RAW_TIMEOUT,
            stream=True,
        )

        if not response.ok:

            print(
                "[github] Failed to download "
                f"{path}: HTTP "
                f"{response.status_code}"
            )

            return False

        first_chunk = True
        contains_binary = False

        with open(
            temporary_destination,
            "wb",
        ) as handle:

            for chunk in response.iter_content(
                chunk_size=64 * 1024
            ):

                if not chunk:
                    continue

                # Check binary content without
                # loading the whole file into RAM.
                if first_chunk:

                    first_chunk = False

                    if b"\x00" in chunk:

                        contains_binary = True
                        break

                handle.write(chunk)

        if contains_binary:

            try:
                os.remove(
                    temporary_destination
                )
            except OSError:
                pass

            return False

        # ----------------------------------------------------
        # UTF-8 VALIDATION
        # ----------------------------------------------------

        try:

            with open(
                temporary_destination,
                "r",
                encoding="utf-8",
            ) as handle:

                # Validate the stream without
                # keeping the complete file in RAM.
                while True:

                    block = handle.read(
                        64 * 1024
                    )

                    if not block:
                        break

        except UnicodeDecodeError:

            try:
                os.remove(
                    temporary_destination
                )
            except OSError:
                pass

            return False

        # ----------------------------------------------------
        # FINALIZE FILE
        # ----------------------------------------------------

        os.replace(
            temporary_destination,
            destination,
        )

        return True

    except requests.RequestException as exc:

        print(
            "[github] Download failed for "
            f"{path}: {exc}"
        )

    except OSError as exc:

        print(
            "[github] File write failed for "
            f"{path}: {exc}"
        )

    finally:

        try:

            if os.path.exists(
                temporary_destination
            ):
                os.remove(
                    temporary_destination
                )

        except OSError:
            pass

    return False


# ============================================================
# DOWNLOAD REPOSITORY
# ============================================================

def download_repo(
    repo_url: str,
    default_branch=None,
):
    """
    Download ALL supported source files.

    Returns:
        (repo_path, temp_root)
    """

    owner, repo = parse_github_url(
        repo_url
    )

    # Resolve default branch if not supplied.
    if not default_branch:

        info = validate_repo(
            repo_url
        )

        default_branch = info[
            "default_branch"
        ]

    # --------------------------------------------------------
    # GET FILE TREE
    # --------------------------------------------------------

    tree = _fetch_repository_tree(
        owner,
        repo,
        default_branch,
    )

    candidates = _candidate_files(
        tree
    )

    if not candidates:

        raise RuntimeError(
            "No supported source files were "
            "found in this repository."
        )

    # --------------------------------------------------------
    # TEMPORARY DIRECTORY
    # --------------------------------------------------------

    temp_root = tempfile.mkdtemp(
        prefix="git_bug_scan_"
    )

    repo_path = os.path.join(
        temp_root,
        "repo",
    )

    os.makedirs(
        repo_path,
        exist_ok=True,
    )

    downloaded = 0
    failed = 0

    try:

        # ----------------------------------------------------
        # DOWNLOAD IN CONTROLLED PARALLEL WORKERS
        # ----------------------------------------------------

        with ThreadPoolExecutor(
            max_workers=DOWNLOAD_WORKERS
        ) as pool:

            future_map = {
                pool.submit(
                    _download_one_file,
                    owner,
                    repo,
                    default_branch,
                    item,
                    repo_path,
                ): item["path"]

                for item in candidates
            }

            for future in as_completed(
                future_map
            ):

                path = future_map[
                    future
                ]

                try:

                    success = future.result()

                    if success:

                        downloaded += 1

                    else:

                        failed += 1

                except Exception as exc:

                    failed += 1

                    print(
                        "[github] Download worker "
                        f"failed for {path}: {exc}"
                    )

        # ----------------------------------------------------
        # FINAL STATUS
        # ----------------------------------------------------

        print(
            "[github] Download complete."
        )

        print(
            f"[github] Selected: "
            f"{len(candidates)}"
        )

        print(
            f"[github] Downloaded: "
            f"{downloaded}"
        )

        print(
            f"[github] Failed/skipped: "
            f"{failed}"
        )

        if downloaded == 0:

            raise RuntimeError(
                "Could not download any supported "
                "source files."
            )

        return (
            repo_path,
            temp_root,
        )

    except Exception:

        shutil.rmtree(
            temp_root,
            ignore_errors=True,
        )

        raise


# ============================================================
# HISTORICAL GITHUB DATA
# ============================================================

def _safe_text(value):
    if value is None:
        return ""

    return str(value).strip()


def _artifact_text(
    artifact: dict
):

    parts = []

    for key, label in (
        ("title", "Title"),
        ("body", "Description"),
        ("state", "State"),
        ("comments", "Comments"),
        ("patch", "Patch"),
    ):

        value = artifact.get(
            key
        )

        if value:

            parts.append(
                f"{label}: {value}"
            )

    labels = artifact.get(
        "labels"
    ) or []

    if labels:

        parts.append(
            "Labels: "
            + ", ".join(labels)
        )

    return "\n".join(parts)


# ============================================================
# PAGINATED GITHUB API
# ============================================================

def _request_paginated(
    url: str,
    max_items: int,
):

    results = []

    page = 1

    per_page = min(
        100,
        max_items,
    )

    while len(results) < max_items:

        try:

            response = requests.get(
                url,
                headers=_github_headers(),
                params={
                    "page": page,
                    "per_page": per_page,
                },
                timeout=GITHUB_TIMEOUT,
            )

        except requests.RequestException as exc:

            print(
                f"[github] Request failed: {exc}"
            )

            break

        if response.status_code == 403:

            print(
                "[github] GitHub API rate limit "
                "may have been reached."
            )

            break

        if not response.ok:

            print(
                "[github] API request failed: "
                f"HTTP {response.status_code}"
            )

            break

        try:

            data = response.json()

        except ValueError:

            break

        if not isinstance(
            data,
            list,
        ):

            break

        if not data:
            break

        results.extend(
            data
        )

        if len(data) < per_page:
            break

        page += 1

    return results[:max_items]


# ============================================================
# HISTORICAL ISSUES
# ============================================================

def fetch_historical_issues(
    owner: str,
    repo: str,
    max_items: int = MAX_ISSUES,
):

    url = (
        f"{GITHUB_API}/repos/"
        f"{owner}/{repo}/issues"
    )

    raw_items = _request_paginated(
        url,
        max_items * 2,
    )

    artifacts = []

    for item in raw_items:

        # GitHub's Issues endpoint can also
        # return pull requests.
        if item.get(
            "pull_request"
        ):
            continue

        labels = [
            label.get("name")

            for label in (
                item.get("labels")
                or []
            )

            if (
                isinstance(
                    label,
                    dict,
                )
                and label.get("name")
            )
        ]

        artifact = {
            "source": "github_issue",

            "artifact_type": "issue",

            "number": item.get(
                "number"
            ),

            "title": _safe_text(
                item.get("title")
            ),

            "body": _safe_text(
                item.get("body")
            ),

            "state": _safe_text(
                item.get("state")
            ),

            "labels": labels,

            "url": item.get(
                "html_url"
            ),

            "created_at": item.get(
                "created_at"
            ),

            "updated_at": item.get(
                "updated_at"
            ),
        }

        artifact["text"] = (
            _artifact_text(
                artifact
            )
        )

        artifacts.append(
            artifact
        )

        if len(artifacts) >= max_items:
            break

    return artifacts


# ============================================================
# HISTORICAL PULL REQUESTS
# ============================================================

def fetch_historical_pull_requests(
    owner: str,
    repo: str,
    max_items: int = MAX_PULL_REQUESTS,
):

    url = (
        f"{GITHUB_API}/repos/"
        f"{owner}/{repo}/pulls"
    )

    raw_items = _request_paginated(
        url,
        max_items,
    )

    artifacts = []

    for item in raw_items:

        number = item.get(
            "number"
        )

        labels = [
            label.get("name")

            for label in (
                item.get("labels")
                or []
            )

            if (
                isinstance(
                    label,
                    dict,
                )
                and label.get("name")
            )
        ]

        patch = ""

        if number is not None:

            patch_url = (
                f"{GITHUB_API}/repos/"
                f"{owner}/{repo}/pulls/"
                f"{number}.patch"
            )

            try:

                response = requests.get(
                    patch_url,
                    headers=_github_headers(),
                    timeout=GITHUB_TIMEOUT,
                )

                if response.ok:

                    patch = (
                        response.text[
                            :30_000
                        ]
                    )

            except requests.RequestException:
                pass

        artifact = {
            "source": "github_pull_request",

            "artifact_type": "pull_request",

            "number": number,

            "title": _safe_text(
                item.get("title")
            ),

            "body": _safe_text(
                item.get("body")
            ),

            "state": _safe_text(
                item.get("state")
            ),

            "labels": labels,

            "url": item.get(
                "html_url"
            ),

            "created_at": item.get(
                "created_at"
            ),

            "updated_at": item.get(
                "updated_at"
            ),

            "patch": patch,
        }

        artifact["text"] = (
            _artifact_text(
                artifact
            )
        )

        artifacts.append(
            artifact
        )

    return artifacts


# ============================================================
# HISTORICAL ARTIFACTS
# ============================================================

def fetch_historical_artifacts(
    repo_url: str,
    max_issues: int = MAX_ISSUES,
    max_pull_requests: int = MAX_PULL_REQUESTS,
):

    owner, repo = parse_github_url(
        repo_url
    )

    print(
        "[github] Retrieving historical "
        f"artifacts for {owner}/{repo}"
    )

    issues = (
        fetch_historical_issues(
            owner,
            repo,
            max_issues,
        )
    )

    pull_requests = (
        fetch_historical_pull_requests(
            owner,
            repo,
            max_pull_requests,
        )
    )

    return {
        "issues": issues,

        "pull_requests": pull_requests,

        "artifacts": (
            issues
            + pull_requests
        ),
    }


# ============================================================
# CLEANUP
# ============================================================

def cleanup_repo(
    temp_root: str
):

    if temp_root:

        shutil.rmtree(
            temp_root,
            ignore_errors=True,
        )


# ============================================================
# FASTAPI COMPATIBILITY FUNCTIONS
# ============================================================

def check_repository(
    repo_url: str
) -> RepoStatus:

    try:

        info = validate_repo(
            repo_url
        )

        return RepoStatus(
            status="valid",

            message=(
                "Repository is valid."
            ),

            owner=info[
                "owner"
            ],

            name=info[
                "repo"
            ],

            default_branch=info[
                "default_branch"
            ],
        )

    except (
        ValueError,
        RuntimeError,
    ) as exc:

        return RepoStatus(
            status="invalid",

            message=str(exc),
        )


def download_repository(
    owner,
    name,
    default_branch,
):

    repo_url = (
        f"https://github.com/"
        f"{owner}/{name}"
    )

    repo_path, temp_root = (
        download_repo(
            repo_url,
            default_branch=default_branch,
        )
    )

    _TEMP_ROOTS[
        repo_path
    ] = temp_root

    return repo_path


def cleanup(
    repo_path
):

    temp_root = (
        _TEMP_ROOTS.pop(
            repo_path,
            None,
        )
    )

    if temp_root:

        cleanup_repo(
            temp_root
        )

    else:

        cleanup_repo(
            repo_path
        )
