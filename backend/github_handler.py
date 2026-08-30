"""
Handles everything related to talking to GitHub:
- validating the URL the user typed
- checking the repo exists / is public
- downloading it to a temp folder so file_scanner.py can read it

Uses plain, unauthenticated requests to the GitHub API - no token needed to
set up. The trade-off is GitHub's public rate limit of 60 requests/hour per
IP, which the error message below explains clearly if it's ever hit.
"""
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


def _describe_403(resp) -> str:
    """
    Turns a GitHub 403 response into a specific, honest reason instead of a
    generic message - a 403 from GitHub can mean a few different things
    (rate limit, temporary block), and each needs a different next step.
    """
    try:
        body = resp.json()
        github_message = body.get("message", "")
    except (ValueError, requests.exceptions.JSONDecodeError):
        github_message = ""

    remaining = resp.headers.get("X-RateLimit-Remaining")
    reset_epoch = resp.headers.get("X-RateLimit-Reset")

    if remaining == "0":
        wait_text = ""
        if reset_epoch:
            try:
                import time
                seconds_left = max(0, int(reset_epoch) - int(time.time()))
                minutes_left = max(1, seconds_left // 60)
                wait_text = f" It resets in about {minutes_left} minute(s)."
            except ValueError:
                pass
        return (
            f"GitHub's rate limit for this app has been used up for now.{wait_text} "
            f"(GitHub said: \"{github_message}\")"
        )

    if "secondary rate limit" in github_message.lower():
        return f"GitHub temporarily blocked this app for making requests too quickly. Please wait a few minutes and try again. (GitHub said: \"{github_message}\")"

    if github_message:
        return f"GitHub rejected this request: \"{github_message}\"."

    return "GitHub rejected this request (403 Forbidden) for an unspecified reason."


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
    except requests.RequestException as e:
        return RepoStatus(
            status="unreachable",
            message=f"Unable to access this repository. Please try again later or check the repository link. (network error: {e})",
        )

    if resp.status_code == 404:
        return RepoStatus(
            status="not_found",
            message="Repository not found. Please check the GitHub URL.",
        )

    if resp.status_code == 403:
        return RepoStatus(
            status="unreachable",
            message=f"Unable to access this repository. {_describe_403(resp)}",
        )

    if resp.status_code != 200:
        return RepoStatus(
            status="unreachable",
            message=f"Unable to access this repository. Please try again later or check the repository link. (GitHub returned status {resp.status_code})",
        )

    data = resp.json()
    if data.get("private"):
        return RepoStatus(
            status="private",
            message="This repository is private and cannot be accessed. Please provide a public repository.",
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
    Downloads the repo as a zip via the GitHub codeload service and
    extracts it to a temp dir. Returns the path to the extracted repo folder.
    """
    tmp_dir = tempfile.mkdtemp(prefix="repo_scan_")
    zip_url = f"https://codeload.github.com/{owner}/{repo}/zip/refs/heads/{branch}"
    zip_path = f"{tmp_dir}/repo.zip"

    resp = requests.get(zip_url, stream=True, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Could not download repository archive (status {resp.status_code}).")

    with open(zip_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    shutil.unpack_archive(zip_path, tmp_dir)
    import os
    os.remove(zip_path)

    extracted = [d for d in os.listdir(tmp_dir) if os.path.isdir(f"{tmp_dir}/{d}")]
    if not extracted:
        raise RuntimeError("Downloaded archive was empty.")
    return f"{tmp_dir}/{extracted[0]}"


def cleanup(path: str):
    import os
    parent = os.path.dirname(path)
    shutil.rmtree(parent, ignore_errors=True)
