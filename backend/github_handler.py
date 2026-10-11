"""
GitHub repository handling.

Responsibilities:
1. Validate GitHub repositories.
2. Download supported source files.
3. Retrieve historical GitHub Issues.
4. Retrieve historical Pull Requests.
5. Retrieve historical PR patches/comments for RAG.
"""

import os
import re
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote

import requests
from dotenv import load_dotenv

from models import RepoStatus

# Load backend/.env when the module is imported.
load_dotenv()

GITHUB_API = "https://api.github.com"
GITHUB_TIMEOUT = 20
RAW_TIMEOUT = 25

MAX_ISSUES = 30
MAX_PULL_REQUESTS = 30
MAX_COMMENTS_PER_ARTIFACT = 20
DOWNLOAD_WORKERS = 8

SOURCE_EXTENSIONS = {
    ".py", ".pyi",
    ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    ".java",
    ".c", ".h", ".cpp", ".hpp", ".cc", ".hh",
    ".cs", ".go", ".php",
    ".rb", ".rs",
    ".kt", ".kts",
    ".swift", ".scala",
    ".sql",
}

SKIP_DIRS = {
    ".git", ".github", ".idea", ".vscode",
    "__pycache__", ".venv", "venv", "env",
    "node_modules", "dist", "build", "coverage",
    ".next", ".nuxt", "vendor", "target",
    "site-packages", "bower_components",
}

_TEMP_ROOTS = {}


# ============================================================
# ENVIRONMENT / GITHUB HEADERS
# ============================================================

def _github_token():
    """Return the configured GitHub token without exposing it."""
    return os.getenv("GITHUB_TOKEN", "").strip()


def _github_headers():
    """Headers for authenticated GitHub API requests."""
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "Git-Bug-Detection-System",
    }

    token = _github_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    return headers


def _raw_headers():
    """
    Headers for raw.githubusercontent.com downloads.

    Authentication is included so private repositories can also be
    downloaded when the token has sufficient permissions.
    """
    headers = {
        "User-Agent": "Git-Bug-Detection-System",
    }

    token = _github_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    return headers


def github_token_status():
    """
    Return safe diagnostic information. Never return the token itself.
    """
    token = _github_token()
    return {
        "configured": bool(token),
        "length": len(token),
        "prefix": (
            token[:4] + "..." if token else ""
        ),
    }


# ============================================================
# PARSE GITHUB URL
# ============================================================

def parse_github_url(repo_url: str):
    if not repo_url:
        raise ValueError("GitHub repository URL is required.")

    value = repo_url.strip()

    # Accept:
    # https://github.com/owner/repo
    # https://github.com/owner/repo.git
    # trailing slash
    match = re.match(
        r"^https?://(?:www\.)?github\.com/([^/\s]+)/([^/#?\s]+?)(?:\.git)?/?$",
        value,
        re.IGNORECASE,
    )

    if not match:
        raise ValueError("Invalid GitHub repository URL.")

    return match.group(1), match.group(2)


# ============================================================
# ERROR DETAILS
# ============================================================

def _github_error_message(response):
    try:
        data = response.json()
        message = data.get("message")
        if message:
            return str(message)
    except (ValueError, requests.RequestException):
        pass

    text = (response.text or "").strip()
    return text[:500] if text else "Unknown GitHub error."


def _raise_github_http_error(response, operation):
    status = response.status_code
    message = _github_error_message(response)

    remaining = response.headers.get("X-RateLimit-Remaining")
    reset = response.headers.get("X-RateLimit-Reset")

    if status == 401:
        raise RuntimeError(
            "GitHub authentication failed (HTTP 401). "
            "Check that GITHUB_TOKEN is valid and loaded from .env."
        )

    if status == 403:
        if remaining == "0":
            raise RuntimeError(
                f"{operation} failed: GitHub API rate limit exceeded "
                f"(HTTP 403). Rate limit reset timestamp: {reset}."
            )

        raise RuntimeError(
            f"{operation} failed: GitHub returned HTTP 403. "
            f"Message: {message}. "
            f"RateLimit-Remaining: {remaining or 'unknown'}."
        )

    if status == 404:
        raise ValueError(
            f"{operation} failed: GitHub repository/resource was not found "
            f"(HTTP 404). Check the URL and token permissions."
        )

    raise RuntimeError(
        f"{operation} failed: HTTP {status}. Message: {message}"
    )


# ============================================================
# VALIDATE REPOSITORY
# ============================================================

def validate_repo(repo_url: str):
    owner, repo = parse_github_url(repo_url)

    url = f"{GITHUB_API}/repos/{owner}/{repo}"

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

    if not response.ok:
        _raise_github_http_error(
            response,
            "GitHub repository validation",
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(
            "GitHub returned an invalid JSON response."
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
        "description": data.get("description"),
        "html_url": data.get(
            "html_url",
            repo_url,
        ),
        "private": bool(data.get("private", False)),
    }


# ============================================================
# PATH HELPERS
# ============================================================

def _should_skip_path(path: str) -> bool:
    parts = path.replace("\\", "/").split("/")

    if any(
        part.lower() in SKIP_DIRS
        for part in parts[:-1]
    ):
        return True

    filename = parts[-1].lower()

    if filename.endswith(
        (".min.js", ".min.css", ".map")
    ):
        return True

    return False


def _is_source_path(path: str) -> bool:
    if _should_skip_path(path):
        return False

    _, ext = os.path.splitext(path)
    return ext.lower() in SOURCE_EXTENSIONS


# ============================================================
# REPOSITORY TREE
# ============================================================

def _fetch_repository_tree(owner: str, repo: str, branch: str):
    branch_encoded = quote(branch, safe="")

    url = (
        f"{GITHUB_API}/repos/{owner}/{repo}/git/trees/"
        f"{branch_encoded}?recursive=1"
    )

    try:
        response = requests.get(
            url,
            headers=_github_headers(),
            timeout=GITHUB_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Unable to retrieve repository file list: {exc}"
        ) from exc

    if not response.ok:
        _raise_github_http_error(
            response,
            "Repository file-list retrieval",
        )

    data = response.json()

    if data.get("truncated"):
        print(
            "[github] GitHub returned a truncated file tree. "
            "Only files included in that tree can be scanned."
        )

    return data.get("tree", [])


# ============================================================
# SOURCE FILE CANDIDATES
# ============================================================

def _candidate_files(tree):
    candidates = []

    for item in tree:
        if item.get("type") != "blob":
            continue

        path = item.get("path", "")
        size = item.get("size")

        if not path or not _is_source_path(path):
            continue

        if not isinstance(size, int) or size <= 0:
            continue

        candidates.append({
            "path": path,
            "size": size,
        })

    total_bytes = sum(item["size"] for item in candidates)

    print(
        f"[github] Selected {len(candidates)} source files "
        f"(estimated {total_bytes / 1_000_000:.1f} MB)."
    )

    return candidates


# ============================================================
# DOWNLOAD ONE SOURCE FILE
# ============================================================

def _download_one_file(owner, repo, branch, item, root_path):
    path = item["path"]

    encoded_path = quote(path, safe="/")
    encoded_branch = quote(branch, safe="/")

    url = (
        "https://raw.githubusercontent.com/"
        f"{owner}/{repo}/{encoded_branch}/{encoded_path}"
    )

    try:
        response = requests.get(
            url,
            headers=_raw_headers(),
            timeout=RAW_TIMEOUT,
        )
    except requests.RequestException as exc:
        print(f"[github] Failed to download {path}: {exc}")
        return False

    if not response.ok:
        print(
            f"[github] Skipping {path}: HTTP "
            f"{response.status_code}"
        )
        return False

    content = response.content

    # Reject binary files.
    if b"\x00" in content:
        return False

    try:
        text_content = content.decode("utf-8")
    except UnicodeDecodeError:
        return False

    destination = os.path.join(root_path, path)
    os.makedirs(
        os.path.dirname(destination),
        exist_ok=True,
    )

    with open(
        destination,
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(text_content)

    return True


# ============================================================
# DOWNLOAD REPOSITORY
# ============================================================

def download_repo(repo_url: str, default_branch=None):
    owner, repo = parse_github_url(repo_url)

    if not default_branch:
        info = validate_repo(repo_url)
        default_branch = info["default_branch"]

    tree = _fetch_repository_tree(
        owner,
        repo,
        default_branch,
    )

    candidates = _candidate_files(tree)

    if not candidates:
        raise RuntimeError(
            "No supported source files were found in this repository."
        )

    temp_root = tempfile.mkdtemp(
        prefix="git_bug_scan_"
    )

    repo_path = os.path.join(
        temp_root,
        "repo",
    )

    os.makedirs(repo_path, exist_ok=True)

    downloaded = 0

    try:
        with ThreadPoolExecutor(
            max_workers=DOWNLOAD_WORKERS
        ) as pool:
            futures = [
                pool.submit(
                    _download_one_file,
                    owner,
                    repo,
                    default_branch,
                    item,
                    repo_path,
                )
                for item in candidates
            ]

            for future in as_completed(futures):
                try:
                    if future.result():
                        downloaded += 1
                except Exception as exc:
                    print(
                        "[github] File download worker failed: "
                        f"{exc}"
                    )

        if downloaded == 0:
            raise RuntimeError(
                "Could not download any supported source files."
            )

        print(
            f"[github] Downloaded {downloaded} source files."
        )

        return repo_path, temp_root

    except Exception:
        shutil.rmtree(
            temp_root,
            ignore_errors=True,
        )
        raise


# ============================================================
# TEXT HELPERS
# ============================================================

def _safe_text(value):
    return "" if value is None else str(value).strip()


def _artifact_text(artifact: dict):
    parts = []

    for key, label in (
        ("title", "Title"),
        ("body", "Description"),
        ("state", "State"),
        ("comments", "Comments"),
        ("patch", "Patch"),
    ):
        value = artifact.get(key)

        if isinstance(value, list):
            value = "\n".join(
                str(item) for item in value
            )

        if value:
            parts.append(f"{label}: {value}")

    labels = artifact.get("labels") or []

    if labels:
        parts.append(
            "Labels: "
            + ", ".join(str(label) for label in labels)
        )

    return "\n".join(parts)


# ============================================================
# PAGINATED GITHUB API REQUEST
# ============================================================

def _request_paginated(url: str, max_items: int, params=None):
    results = []
    page = 1

    if max_items <= 0:
        return results

    per_page = min(100, max_items)
    base_params = dict(params or {})

    while len(results) < max_items:
        request_params = {
            **base_params,
            "page": page,
            "per_page": per_page,
        }

        try:
            response = requests.get(
                url,
                headers=_github_headers(),
                params=request_params,
                timeout=GITHUB_TIMEOUT,
            )
        except requests.RequestException as exc:
            print(f"[github] Request failed: {exc}")
            break

        if response.status_code in (401, 403, 404):
            try:
                _raise_github_http_error(
                    response,
                    "GitHub API request",
                )
            except (ValueError, RuntimeError) as exc:
                print(f"[github] {exc}")
            break

        if not response.ok:
            print(
                "[github] API request failed: "
                f"HTTP {response.status_code}"
            )
            print((response.text or "")[:500])
            break

        try:
            data = response.json()
        except ValueError:
            print("[github] API returned invalid JSON.")
            break

        if not isinstance(data, list) or not data:
            break

        results.extend(data)

        if len(data) < per_page:
            break

        page += 1

    return results[:max_items]


# ============================================================
# ISSUE / PR COMMENTS
# ============================================================

def _fetch_issue_comments(owner: str, repo: str, number: int):
    url = (
        f"{GITHUB_API}/repos/{owner}/{repo}/issues/"
        f"{number}/comments"
    )

    raw_comments = _request_paginated(
        url,
        MAX_COMMENTS_PER_ARTIFACT,
    )

    comments = []

    for item in raw_comments:
        body = _safe_text(item.get("body"))
        if body:
            comments.append(body)

    return comments


def _fetch_pull_request_comments(
    owner: str,
    repo: str,
    number: int,
):
    # PR conversation comments are available through
    # the issues comments endpoint.
    return _fetch_issue_comments(owner, repo, number)


# ============================================================
# HISTORICAL ISSUES
# ============================================================

def fetch_historical_issues(
    owner: str,
    repo: str,
    max_items: int = MAX_ISSUES,
):
    url = f"{GITHUB_API}/repos/{owner}/{repo}/issues"

    raw_items = _request_paginated(
        url,
        max_items * 2,
        params={
            "state": "all",
            "sort": "updated",
            "direction": "desc",
        },
    )

    artifacts = []

    for item in raw_items:
        # GitHub returns PRs through the Issues endpoint.
        if item.get("pull_request"):
            continue

        number = item.get("number")

        labels = [
            label.get("name")
            for label in item.get("labels", [])
            if isinstance(label, dict) and label.get("name")
        ]

        comments = (
            _fetch_issue_comments(
                owner,
                repo,
                number,
            )
            if number is not None
            else []
        )

        artifact = {
            "source": "github_issue",
            "artifact_type": "issue",
            "number": number,
            "title": _safe_text(item.get("title")),
            "body": _safe_text(item.get("body")),
            "state": _safe_text(item.get("state")),
            "labels": labels,
            "comments": comments,
            "url": item.get("html_url"),
            "created_at": item.get("created_at"),
            "updated_at": item.get("updated_at"),
            "closed_at": item.get("closed_at"),
        }

        artifact["text"] = _artifact_text(artifact)
        artifacts.append(artifact)

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
    url = f"{GITHUB_API}/repos/{owner}/{repo}/pulls"

    raw_items = _request_paginated(
        url,
        max_items * 2,
        params={
            "state": "all",
            "sort": "updated",
            "direction": "desc",
        },
    )

    artifacts = []

    for item in raw_items:
        number = item.get("number")

        labels = [
            label.get("name")
            for label in item.get("labels", [])
            if isinstance(label, dict) and label.get("name")
        ]

        patch = ""

        if number is not None:
            patch_url = (
                f"{GITHUB_API}/repos/{owner}/{repo}/pulls/"
                f"{number}.patch"
            )

            try:
                response = requests.get(
                    patch_url,
                    headers=_github_headers(),
                    timeout=GITHUB_TIMEOUT,
                )

                if response.ok:
                    patch = response.text[:30000]
                else:
                    print(
                        f"[github] Could not retrieve patch "
                        f"for PR #{number}: HTTP "
                        f"{response.status_code}"
                    )
            except requests.RequestException as exc:
                print(
                    f"[github] PR patch request failed: {exc}"
                )

        comments = (
            _fetch_pull_request_comments(
                owner,
                repo,
                number,
            )
            if number is not None
            else []
        )

        merged_at = item.get("merged_at")

        artifact = {
            "source": "github_pull_request",
            "artifact_type": "pull_request",
            "number": number,
            "title": _safe_text(item.get("title")),
            "body": _safe_text(item.get("body")),
            "state": _safe_text(item.get("state")),
            "merged": merged_at is not None,
            "merged_at": merged_at,
            "labels": labels,
            "comments": comments,
            "url": item.get("html_url"),
            "created_at": item.get("created_at"),
            "updated_at": item.get("updated_at"),
            "patch": patch,
        }

        artifact["text"] = _artifact_text(artifact)
        artifacts.append(artifact)

        if len(artifacts) >= max_items:
            break

    # Put merged PRs first because their patches represent
    # actual historical fixes.
    artifacts.sort(
        key=lambda item: (
            not bool(item.get("merged")),
            item.get("updated_at") or "",
        )
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
    owner, repo = parse_github_url(repo_url)

    print(
        f"[github] Retrieving historical artifacts "
        f"for {owner}/{repo}"
    )

    issues = fetch_historical_issues(
        owner,
        repo,
        max_issues,
    )

    pull_requests = fetch_historical_pull_requests(
        owner,
        repo,
        max_pull_requests,
    )

    artifacts = issues + pull_requests

    print(
        "[github] Historical retrieval complete: "
        f"{len(issues)} issues + "
        f"{len(pull_requests)} PRs = "
        f"{len(artifacts)} artifacts"
    )

    return {
        "issues": issues,
        "pull_requests": pull_requests,
        "artifacts": artifacts,
    }


# ============================================================
# CLEANUP
# ============================================================

def cleanup_repo(temp_root: str):
    if temp_root:
        shutil.rmtree(
            temp_root,
            ignore_errors=True,
        )


# ============================================================
# FASTAPI REPOSITORY CHECK
# ============================================================

def check_repository(repo_url: str) -> RepoStatus:
    try:
        info = validate_repo(repo_url)

        return RepoStatus(
            status="valid",
            message="Repository is valid.",
            owner=info["owner"],
            name=info["repo"],
            default_branch=info["default_branch"],
        )

    except (ValueError, RuntimeError) as exc:
        return RepoStatus(
            status="invalid",
            message=str(exc),
        )


# ============================================================
# FASTAPI DOWNLOAD WRAPPER
# ============================================================

def download_repository(owner, name, default_branch):
    repo_url = f"https://github.com/{owner}/{name}"

    repo_path, temp_root = download_repo(
        repo_url,
        default_branch=default_branch,
    )

    _TEMP_ROOTS[repo_path] = temp_root

    return repo_path


# ============================================================
# FASTAPI CLEANUP WRAPPER
# ============================================================

def cleanup(repo_path):
    temp_root = _TEMP_ROOTS.pop(
        repo_path,
        None,
    )

    if temp_root:
        cleanup_repo(temp_root)
    else:
        cleanup_repo(repo_path)
