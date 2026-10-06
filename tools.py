import requests

def get_issue(owner: str, repo: str, issue_number: int) -> dict:
    """Fetch an issue by Owner, repo, and issue id."""
    response = requests.get(
        f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}",
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def get_issues_by_repo(owner: str, repo: str, state: str = "open") -> dict:
    """Fetches all the issues in a repo"""
    response = requests.get(
        f"https://api.github.com/repos/{owner}/{repo}/issues?state={state}",
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def get_comments_on_issue(owner: str, repo: str, issue_number: int) -> dict:
    """Fetches all the comments for an issue in a repo"""
    response = requests.get(
        f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/comments",
        timeout=10,
    )
    response.raise_for_status()
    return response.json()

def get_languages_in_repo(owner: str, repo: str) -> dict:
    """Fetches all languages used in a repo"""
    response = requests.get(
        f"https://api.github.com/repos/{owner}/{repo}/languages",
        timeout=10,
    )
    response.raise_for_status()
    return response.json()
