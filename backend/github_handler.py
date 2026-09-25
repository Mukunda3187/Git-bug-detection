
"""
GitHub repository handling.

Downloads a bounded selection of source files from a repository instead
of downloading and extracting the entire repository ZIP.

Also retrieves historical GitHub Issues and Pull Requests for RAG.
"""

import os
import re
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote

import requests

from models import RepoStatus


GITHUB_API = "https://api.github.com"
GITHUB_TIMEOUT = 20
RAW_TIMEOUT = 25

MAX_ISSUES = 30
MAX_PULL_REQUESTS = 30

# Resource limits for repository scanning.
MAX_SOURCE_FILES = 250
MAX_FILE_BYTES = 300_000
MAX_TOTAL_BYTES = 12_000_000
DOWNLOAD_WORKERS = 8

SOURCE_EXTENSIONS = {
    ".py", ".pyi",
    ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    ".java", ".c", ".h", ".cpp", ".hpp", ".cc", ".hh",
    ".cs", ".go", ".php", ".rb", ".rs", ".kt", ".kts",
    ".swift", ".scala", ".sql",
}

SKIP_DIRS = {
    ".git", ".github", ".idea", ".vscode",
    "__pycache__", ".venv", "venv", "env",
    "node_modules", "dist", "build", "coverage",
    ".next", ".nuxt", "vendor", "target",
    "site-packages", "bower_components",
}

_TEMP_ROOTS = {}


def _github_headers():
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    return headers


def parse_github_url(repo_url: str):
    if not repo_url:
        raise ValueError("GitHub repository URL is required.")

    match = re.match(
        r"^https?://(?:www\.)?github\.com/([^/]+)/([^/#?]+?)(?:\.git)?/?$",
        repo_url.strip(),
        re.IGNORECASE,
    )

    if not match:
        raise ValueError("Invalid GitHub repository URL.")

    return match.group(1), match.group(2)


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
        raise RuntimeError(f"Unable to contact GitHub: {exc}") from exc

    if response.status_code == 404:
        raise ValueError("GitHub repository not found.")

    if not response.ok:
        raise RuntimeError(
            f"GitHub repository validation failed: HTTP {response.status_code}"
        )

    data = response.json()

    return {
        "owner": owner,
        "repo": repo,
        "full_name": data.get("full_name", f"{owner}/{repo}"),
        "default_branch": data.get("default_branch", "main"),
        "description": data.get("description"),
        "html_url": data.get("html_url", repo_url),
    }


def _should_skip_path(path: str) -> bool:
    parts = path.replace("\\", "/").split("/")

    if any(part.lower() in SKIP_DIRS for part in parts[:-1]):
        return True

    filename = parts[-1].lower()

    if filename.endswith((".min.js", ".min.css", ".map")):
        return True

    return False


def _is_source_path(path: str) -> bool:
    if _should_skip_path(path):
        return False

    _, ext = os.path.splitext(path)
    return ext.lower() in SOURCE_EXTENSIONS


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
        raise RuntimeError(f"Unable to retrieve repository file list: {exc}") from exc

    if not response.ok:
        raise RuntimeError(
            f"Unable to retrieve repository file list: HTTP {response.status_code}"
        )

    data = response.json()

    if data.get("truncated"):
        print(
            "[github] GitHub returned a truncated file tree. "
            "Only the files included in that tree can be scanned."
        )

    return data.get("tree", [])


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

        if size > MAX_FILE_BYTES:
            continue

        candidates.append({
            "path": path,
            "size": size,
        })

    # Prefer normal application source over test files when the limit is hit.
    def sort_key(item):
        path = item["path"].lower()
        is_test = (
            "/test/" in f"/{path}/"
            or "/tests/" in f"/{path}/"
            or path.startswith(("test_", "tests/"))
        )
        return (is_test, path.count("/"), path)

    candidates.sort(key=sort_key)

    selected = []
    total_bytes = 0

    for item in candidates:
        if len(selected) >= MAX_SOURCE_FILES:
            break

        if total_bytes + item["size"] > MAX_TOTAL_BYTES:
            continue

        selected.append(item)
        total_bytes += item["size"]

    print(
        f"[github] Selected {len(selected)} source files "
        f"(estimated {total_bytes / 1_000_000:.1f} MB)."
    )

    return selected


def _download_one_file(owner, repo, branch, item, root_path):
    path = item["path"]
    encoded_path = quote(path, safe="/")
    encoded_branch = quote(branch, safe="/")

    url = (
        f"https://raw.githubusercontent.com/{owner}/{repo}/"
        f"{encoded_branch}/{encoded_path}"
    )

    try:
        response = requests.get(url, timeout=RAW_TIMEOUT)
    except requests.RequestException as exc:
        print(f"[github] Failed to download {path}: {exc}")
        return False

    if not response.ok:
        print(
            f"[github] Skipping {path}: HTTP {response.status_code}"
        )
        return False

    content = response.content

    if len(content) > MAX_FILE_BYTES:
        print(f"[github] Skipping oversized file: {path}")
        return False

    # Reject binary content.
    if b"\x00" in content:
        return False

    try:
        text_content = content.decode("utf-8")
    except UnicodeDecodeError:
        return False

    destination = os.path.join(root_path, path)
    os.makedirs(os.path.dirname(destination), exist_ok=True)

    with open(destination, "w", encoding="utf-8") as handle:
        handle.write(text_content)

    return True


def download_repo(repo_url: str, default_branch=None):
    """
    Download a bounded set of source files.

    Returns:
        (repo_path, temp_root)
    """
    owner, repo = parse_github_url(repo_url)

    if not default_branch:
        info = validate_repo(repo_url)
        default_branch = info["default_branch"]

    tree = _fetch_repository_tree(owner, repo, default_branch)
    candidates = _candidate_files(tree)

    if not candidates:
        raise RuntimeError(
            "No supported source files were found in this repository."
        )

    temp_root = tempfile.mkdtemp(prefix="git_bug_scan_")
    repo_path = os.path.join(temp_root, "repo")
    os.makedirs(repo_path, exist_ok=True)

    downloaded = 0

    try:
        with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
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
                    print(f"[github] File download worker failed: {exc}")

        if downloaded == 0:
            raise RuntimeError(
                "Could not download any supported source files."
            )

        print(f"[github] Downloaded {downloaded} source files.")

        return repo_path, temp_root

    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise


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
        if value:
            parts.append(f"{label}: {value}")

    labels = artifact.get("labels") or []
    if labels:
        parts.append("Labels: " + ", ".join(labels))

    return "\n".join(parts)


def _request_paginated(url: str, max_items: int):
    results = []
    page = 1
    per_page = min(100, max_items)

    while len(results) < max_items:
        try:
            response = requests.get(
                url,
                headers=_github_headers(),
                params={"page": page, "per_page": per_page},
                timeout=GITHUB_TIMEOUT,
            )
        except requests.RequestException as exc:
            print(f"[github] Request failed: {exc}")
            break

        if response.status_code == 403:
            print("[github] GitHub API rate limit may have been reached.")
            break

        if not response.ok:
            print(f"[github] API request failed: HTTP {response.status_code}")
            break

        data = response.json()
        if not isinstance(data, list) or not data:
            break

        results.extend(data)

        if len(data) < per_page:
            break

        page += 1

    return results[:max_items]


def fetch_historical_issues(
    owner: str,
    repo: str,
    max_items: int = MAX_ISSUES,
):
    url = f"{GITHUB_API}/repos/{owner}/{repo}/issues"
    raw_items = _request_paginated(url, max_items * 2)
    artifacts = []

    for item in raw_items:
        if item.get("pull_request"):
            continue

        labels = [
            label.get("name")
            for label in item.get("labels", [])
            if isinstance(label, dict) and label.get("name")
        ]

        artifact = {
            "source": "github_issue",
            "artifact_type": "issue",
            "number": item.get("number"),
            "title": _safe_text(item.get("title")),
            "body": _safe_text(item.get("body")),
            "state": _safe_text(item.get("state")),
            "labels": labels,
            "url": item.get("html_url"),
            "created_at": item.get("created_at"),
            "updated_at": item.get("updated_at"),
        }
        artifact["text"] = _artifact_text(artifact)
        artifacts.append(artifact)

        if len(artifacts) >= max_items:
            break

    return artifacts


def fetch_historical_pull_requests(
    owner: str,
    repo: str,
    max_items: int = MAX_PULL_REQUESTS,
):
    url = f"{GITHUB_API}/repos/{owner}/{repo}/pulls"
    raw_items = _request_paginated(url, max_items)
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
            patch_url = f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{number}.patch"

            try:
                response = requests.get(
                    patch_url,
                    headers=_github_headers(),
                    timeout=GITHUB_TIMEOUT,
                )
                if response.ok:
                    patch = response.text[:30_000]
            except requests.RequestException:
                pass

        artifact = {
            "source": "github_pull_request",
            "artifact_type": "pull_request",
            "number": number,
            "title": _safe_text(item.get("title")),
            "body": _safe_text(item.get("body")),
            "state": _safe_text(item.get("state")),
            "labels": labels,
            "url": item.get("html_url"),
            "created_at": item.get("created_at"),
            "updated_at": item.get("updated_at"),
            "patch": patch,
        }
        artifact["text"] = _artifact_text(artifact)
        artifacts.append(artifact)

    return artifacts


def fetch_historical_artifacts(
    repo_url: str,
    max_issues: int = MAX_ISSUES,
    max_pull_requests: int = MAX_PULL_REQUESTS,
):
    owner, repo = parse_github_url(repo_url)

    print(f"[github] Retrieving historical artifacts for {owner}/{repo}")

    issues = fetch_historical_issues(owner, repo, max_issues)
    pull_requests = fetch_historical_pull_requests(
        owner, repo, max_pull_requests
    )

    return {
        "issues": issues,
        "pull_requests": pull_requests,
        "artifacts": issues + pull_requests,
    }


def cleanup_repo(temp_root: str):
    if temp_root:
        shutil.rmtree(temp_root, ignore_errors=True)


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


def download_repository(owner, name, default_branch):
    repo_url = f"https://github.com/{owner}/{name}"

    repo_path, temp_root = download_repo(
        repo_url,
        default_branch=default_branch,
    )

    _TEMP_ROOTS[repo_path] = temp_root
    return repo_path


def cleanup(repo_path):
    temp_root = _TEMP_ROOTS.pop(repo_path, None)

    if temp_root:
        cleanup_repo(temp_root)
    else:
        cleanup_repo(repo_path)
