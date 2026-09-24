"""
GitHub repository handling and historical artifact retrieval.

This module:
1. Validates GitHub repository URLs.
2. Downloads repositories for source scanning.
3. Retrieves historical GitHub Issues and Pull Requests.
4. Converts historical artifacts into RAG-ready records.
"""

import os
import re
import shutil
import tempfile
import zipfile

import requests


GITHUB_API = "https://api.github.com"

GITHUB_TIMEOUT = 20

MAX_ISSUES = 30
MAX_PULL_REQUESTS = 30


def _github_headers():
    """
    Create GitHub API headers.

    A GITHUB_TOKEN is optional. When supplied, GitHub API
    rate limits are higher.
    """

    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    token = os.getenv(
        "GITHUB_TOKEN"
    )

    if token:
        headers[
            "Authorization"
        ] = f"Bearer {token}"

    return headers


def parse_github_url(
    repo_url: str,
):
    """
    Extract owner and repository name from a GitHub URL.

    Returns:
        (owner, repo)

    Raises:
        ValueError for invalid GitHub URLs.
    """

    if not repo_url:
        raise ValueError(
            "GitHub repository URL is required."
        )

    url = repo_url.strip()

    pattern = (
        r"^https?://"
        r"(?:www\.)?"
        r"github\.com/"
        r"([^/]+)/"
        r"([^/#?]+)"
        r"(?:/)?$"
    )

    match = re.match(
        pattern,
        url,
        re.IGNORECASE,
    )

    if not match:

        raise ValueError(
            "Invalid GitHub repository URL."
        )

    owner = match.group(
        1
    )

    repo = match.group(
        2
    )

    if repo.endswith(
        ".git"
    ):

        repo = repo[:-4]

    return owner, repo


def validate_repo(
    repo_url: str,
):
    """
    Validate that a GitHub repository exists.

    Returns:
        Dictionary containing repository information.
    """

    owner, repo = parse_github_url(
        repo_url
    )

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

    except requests.RequestException as e:

        raise RuntimeError(
            f"Unable to contact GitHub: {e}"
        )

    if response.status_code == 404:

        raise ValueError(
            "GitHub repository not found."
        )

    if not response.ok:

        raise RuntimeError(
            "GitHub repository validation failed: "
            f"HTTP {response.status_code}"
        )

    data = response.json()

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


def download_repo(
    repo_url: str,
):
    """
    Download a GitHub repository as a ZIP archive
    and extract it into a temporary directory.

    Returns:
        (repo_path, temp_root)
    """

    owner, repo = parse_github_url(
        repo_url
    )

    api_url = (
        f"{GITHUB_API}/repos/"
        f"{owner}/{repo}"
    )

    try:

        response = requests.get(
            api_url,
            headers=_github_headers(),
            timeout=GITHUB_TIMEOUT,
        )

    except requests.RequestException as e:

        raise RuntimeError(
            f"Unable to contact GitHub: {e}"
        )

    if not response.ok:

        raise RuntimeError(
            "Unable to retrieve repository: "
            f"HTTP {response.status_code}"
        )

    repo_data = response.json()

    default_branch = repo_data.get(
        "default_branch",
        "main",
    )

    archive_url = (
        f"{GITHUB_API}/repos/"
        f"{owner}/{repo}/zipball/"
        f"{default_branch}"
    )

    try:

        archive_response = requests.get(
            archive_url,
            headers=_github_headers(),
            timeout=60,
        )

    except requests.RequestException as e:

        raise RuntimeError(
            f"Unable to download repository: {e}"
        )

    if not archive_response.ok:

        raise RuntimeError(
            "Repository download failed: "
            f"HTTP {archive_response.status_code}"
        )

    temp_root = tempfile.mkdtemp(
        prefix="git_bug_scan_"
    )

    archive_path = os.path.join(
        temp_root,
        "repository.zip",
    )

    with open(
        archive_path,
        "wb",
    ) as f:

        f.write(
            archive_response.content
        )

    extract_path = os.path.join(
        temp_root,
        "repo",
    )

    os.makedirs(
        extract_path,
        exist_ok=True,
    )

    try:

        with zipfile.ZipFile(
            archive_path,
            "r",
        ) as archive:

            archive.extractall(
                extract_path
            )

    except Exception:

        shutil.rmtree(
            temp_root,
            ignore_errors=True,
        )

        raise

    entries = os.listdir(
        extract_path
    )

    if len(entries) == 1:

        possible_root = os.path.join(
            extract_path,
            entries[0],
        )

        if os.path.isdir(
            possible_root
        ):

            repo_path = possible_root

        else:

            repo_path = extract_path

    else:

        repo_path = extract_path

    return repo_path, temp_root


def _safe_text(
    value,
):
    """
    Convert a GitHub field to safe text.
    """

    if value is None:
        return ""

    return str(
        value
    ).strip()


def _artifact_text(
    artifact: dict,
):
    """
    Build the text representation used for RAG.
    """

    parts = []

    if artifact.get(
        "title"
    ):

        parts.append(
            f"Title: {artifact['title']}"
        )

    if artifact.get(
        "body"
    ):

        parts.append(
            f"Description: {artifact['body']}"
        )

    if artifact.get(
        "state"
    ):

        parts.append(
            f"State: {artifact['state']}"
        )

    if artifact.get(
        "labels"
    ):

        labels = ", ".join(
            artifact["labels"]
        )

        if labels:

            parts.append(
                f"Labels: {labels}"
            )

    if artifact.get(
        "comments"
    ):

        parts.append(
            f"Comments: {artifact['comments']}"
        )

    if artifact.get(
        "patch"
    ):

        parts.append(
            f"Patch: {artifact['patch']}"
        )

    return "\n".join(
        filter(
            None,
            parts,
        )
    )


def _request_paginated(
    url: str,
    max_items: int,
):
    """
    Retrieve GitHub API pages until max_items is reached.
    """

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

        except requests.RequestException as e:

            print(
                f"[github] Request failed: {e}"
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

        data = response.json()

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

    return results[
        :max_items
    ]


def fetch_historical_issues(
    owner: str,
    repo: str,
    max_items: int = MAX_ISSUES,
):
    """
    Retrieve historical GitHub issues.

    Pull requests are excluded because GitHub's issues endpoint
    also returns pull-request entries.
    """

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

        # GitHub represents pull requests inside the issues API.
        if item.get(
            "pull_request"
        ):

            continue

        labels = []

        for label in item.get(
            "labels",
            [],
        ):

            if isinstance(
                label,
                dict,
            ):

                name = label.get(
                    "name"
                )

                if name:
                    labels.append(
                        name
                    )

        artifact = {
            "source": "github_issue",
            "artifact_type": "issue",
            "number": item.get(
                "number"
            ),
            "title": _safe_text(
                item.get(
                    "title"
                )
            ),
            "body": _safe_text(
                item.get(
                    "body"
                )
            ),
            "state": _safe_text(
                item.get(
                    "state"
                )
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

        artifact[
            "text"
        ] = _artifact_text(
            artifact
        )

        artifacts.append(
            artifact
        )

        if len(artifacts) >= max_items:
            break

    return artifacts


def fetch_historical_pull_requests(
    owner: str,
    repo: str,
    max_items: int = MAX_PULL_REQUESTS,
):
    """
    Retrieve historical GitHub pull requests.

    Patch information is retrieved when possible so the RAG system
    can use both the PR description and its code-change context.
    """

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

        labels = []

        for label in item.get(
            "labels",
            [],
        ):

            if isinstance(
                label,
                dict,
            ):

                name = label.get(
                    "name"
                )

                if name:
                    labels.append(
                        name
                    )

        patch = ""

        if number is not None:

            patch_url = (
                f"{GITHUB_API}/repos/"
                f"{owner}/{repo}/pulls/"
                f"{number}.patch"
            )

            try:

                patch_response = requests.get(
                    patch_url,
                    headers=_github_headers(),
                    timeout=GITHUB_TIMEOUT,
                )

                if patch_response.ok:

                    patch = patch_response.text[
                        :30000
                    ]

            except requests.RequestException:

                patch = ""

        artifact = {
            "source": "github_pull_request",
            "artifact_type": "pull_request",
            "number": number,
            "title": _safe_text(
                item.get(
                    "title"
                )
            ),
            "body": _safe_text(
                item.get(
                    "body"
                )
            ),
            "state": _safe_text(
                item.get(
                    "state"
                )
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

        artifact[
            "text"
        ] = _artifact_text(
            artifact
        )

        artifacts.append(
            artifact
        )

    return artifacts


def fetch_historical_artifacts(
    repo_url: str,
    max_issues: int = MAX_ISSUES,
    max_pull_requests: int = MAX_PULL_REQUESTS,
):
    """
    Retrieve both historical Issues and Pull Requests.

    Returns:
        {
            "issues": [...],
            "pull_requests": [...],
            "artifacts": [...]
        }
    """

    owner, repo = parse_github_url(
        repo_url
    )

    print(
        f"[github] Retrieving historical artifacts "
        f"for {owner}/{repo}"
    )

    issues = fetch_historical_issues(
        owner,
        repo,
        max_issues,
    )

    pull_requests = (
        fetch_historical_pull_requests(
            owner,
            repo,
            max_pull_requests,
        )
    )

    artifacts = (
        issues
        + pull_requests
    )

    print(
        f"[github] Retrieved "
        f"{len(issues)} issues and "
        f"{len(pull_requests)} pull requests."
    )

    return {
        "issues": issues,
        "pull_requests": pull_requests,
        "artifacts": artifacts,
    }


def cleanup_repo(
    temp_root: str,
):
    """
    Remove a downloaded temporary repository.
    """

    if not temp_root:
        return

    try:

        shutil.rmtree(
            temp_root,
            ignore_errors=True,
        )

    except Exception as e:

        print(
            f"[github] Cleanup failed: {e}"
        )
