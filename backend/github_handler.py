"""
Handles everything related to talking to GitHub:
- validating the URL the user typed
- checking the repo exists / is public
- downloading it to a temp folder so file_scanner.py can read it
"""
import os
import re
import shutil
import tempfile
import requests
from models import RepoStatus

GITHUB_URL_RE = re.compile(
    r"^https?://github\.com/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+?)(\.git)?/?$"
)


def parse_github_url(url: str):
    """Returns (owner, repo) or (None, None) if the URL doesn't look like a GitHub repo URL."""
    if not url:
        return None, None
    match = GITHUB_URL_RE.match(url.strip())
    if not match:
        return None, None
    return match.group("owner"), match.group("repo")


def check_repository(url: str) -> RepoStatus:
    """
    Validates the URL, then asks the GitHub REST API whether the repo exists
    and is public. Never raises - always returns a RepoStatus the frontend
    can render directly.
    """
    owner, repo = parse_github_url(url)
    if not owner or not repo:
        return RepoStatus(
            status="invalid",
            message="Invalid GitHub repository link. Please enter a valid GitHub repository URL.",
        )

    api_url = f"https://api.github.com/repos/{owner}/{repo}"
    try:
        resp = requests.get(api_url, timeout=10)
    except requests.RequestException:
        return RepoStatus(
            status="unreachable",
            message="Unable to access this repository. Please try again later or check the repository link.",
        )

    if resp.status_code == 404:
        return RepoStatus(
            status="not_found",
            message="Repository not found. Please check the GitHub URL.",
        )

    if resp.status_code != 200:
        return RepoStatus(
            status="unreachable",
            message="Unable to access this repository. Please try again later or check the repository link.",
        )

    data = resp.json()
    if data.get("private"):
        return RepoStatus(
            status="private",
            message="This repository is private and cannot be accessed. Please provide a public repository or configure authentication.",
        )

    return RepoStatus(
        status="valid",
        message="Repository is valid and accessible.",
        owner=owner,
        name=repo,
        default_branch=data.get("default_branch", "main"),
    )


def download_repository(owner: str, repo: str, branch: str) -> str:
    """
    Downloads the repo as a zip via the GitHub codeload service (no git
    binary or auth needed for public repos) and extracts it to a temp dir.
    Returns the path to the extracted repo folder.
    """
    tmp_dir = tempfile.mkdtemp(prefix="repo_scan_")
    zip_url = f"https://codeload.github.com/{owner}/{repo}/zip/refs/heads/{branch}"
    zip_path = os.path.join(tmp_dir, "repo.zip")

    resp = requests.get(zip_url, stream=True, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError("Could not download repository archive.")

    with open(zip_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    shutil.unpack_archive(zip_path, tmp_dir)
    os.remove(zip_path)

    # The zip extracts into a single folder like "repo-branch/"
    extracted = [d for d in os.listdir(tmp_dir) if os.path.isdir(os.path.join(tmp_dir, d))]
    if not extracted:
        raise RuntimeError("Downloaded archive was empty.")
    return os.path.join(tmp_dir, extracted[0])


def cleanup(path: str):
    parent = os.path.dirname(path)
    shutil.rmtree(parent, ignore_errors=True)
