"""
GitHub PR Outcome Discovery
Crawls GitHub repositories to collect (review_comment, resulting_diff, merge_outcome) triples.
These are the core training pairs for MergePilot.

Quality filters:
  - PR must be merged (closed + merged = True)
  - At least 2 reviewer approvals
  - At least 1 test file in the diff
  - Diff under 400 lines (scope discipline signal)
  - Review comment must have triggered a code change (not just acknowledged)
"""

import json
import os
import time
from pathlib import Path
from typing import Iterator

import httpx
from loguru import logger
from tqdm import tqdm


# Quality thresholds
QUALITY_FILTERS = {
    "min_approvals": 2,
    "max_diff_lines": 400,
    "must_have_tests": True,
    "must_be_merged": True,
}

SUPPORTED_LANGUAGES = {"Python", "TypeScript", "JavaScript", "Go", "Rust", "Java"}


def get_github_headers(token: str) -> dict:
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def get_top_repos(
    client: httpx.Client,
    token: str,
    n_repos: int = 50000,
    min_stars: int = 500,
) -> list[dict]:
    """Fetch top N repositories by stars across supported languages."""
    headers = get_github_headers(token)
    repos = []

    per_language = max(1, n_repos // len(SUPPORTED_LANGUAGES))

    for language in SUPPORTED_LANGUAGES:
        page = 1
        language_count = 0
        while language_count < per_language:
            resp = client.get(
                "https://api.github.com/search/repositories",
                params={
                    "q": f"language:{language} stars:>={min_stars} is:public archived:false",
                    "sort": "stars",
                    "order": "desc",
                    "per_page": 100,
                    "page": page,
                },
                headers=headers,
                timeout=30,
            )
            if resp.status_code == 403:
                wait = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60)) - int(time.time())
                logger.warning(f"Rate limited, waiting {wait}s...")
                time.sleep(max(wait + 5, 5))
                continue
            if resp.status_code != 200:
                logger.warning(f"Search failed for {language}: {resp.status_code}")
                break

            data = resp.json()
            batch = data.get("items", [])
            if not batch:
                break

            for r in batch:
                repos.append({
                    "full_name": r["full_name"],
                    "language": r.get("language"),
                    "stars": r.get("stargazers_count", 0),
                    "default_branch": r.get("default_branch", "main"),
                })
                language_count += 1
                if language_count >= per_language:
                    break

            if len(batch) < 100:
                break
            page += 1
            time.sleep(0.5)  # Respect rate limits

    logger.info(f"Discovered {len(repos)} repositories")
    return repos[:n_repos]


def get_merged_prs(
    client: httpx.Client,
    token: str,
    repo: str,
    max_prs: int = 100,
) -> list[dict]:
    """Fetch merged PRs for a single repository."""
    headers = get_github_headers(token)
    prs = []
    page = 1

    while len(prs) < max_prs:
        resp = client.get(
            f"https://api.github.com/repos/{repo}/pulls",
            params={"state": "closed", "per_page": 50, "page": page},
            headers=headers,
            timeout=30,
        )
        if resp.status_code != 200:
            break
        batch = resp.json()
        if not batch:
            break

        for pr in batch:
            if pr.get("merged_at"):
                prs.append({
                    "number": pr["number"],
                    "title": pr.get("title", ""),
                    "body": pr.get("body", ""),
                    "merged_at": pr.get("merged_at"),
                })
        page += 1
        time.sleep(0.1)

    return prs[:max_prs]


def get_pr_reviews(
    client: httpx.Client,
    token: str,
    repo: str,
    pr_number: int,
) -> list[dict]:
    """Fetch review comments for a PR."""
    headers = get_github_headers(token)
    resp = client.get(
        f"https://api.github.com/repos/{repo}/pulls/{pr_number}/reviews",
        headers=headers,
        timeout=30,
    )
    if resp.status_code != 200:
        return []
    return resp.json()


def get_pr_diff(
    client: httpx.Client,
    token: str,
    repo: str,
    pr_number: int,
) -> str:
    """Fetch the unified diff for a PR."""
    headers = {**get_github_headers(token), "Accept": "application/vnd.github.diff"}
    resp = client.get(
        f"https://api.github.com/repos/{repo}/pulls/{pr_number}",
        headers=headers,
        timeout=30,
    )
    if resp.status_code != 200:
        return ""
    return resp.text


def get_pr_review_comments(
    client: httpx.Client,
    token: str,
    repo: str,
    pr_number: int,
) -> list[dict]:
    """Fetch inline review comments for a PR."""
    headers = get_github_headers(token)
    comments = []
    page = 1

    while True:
        resp = client.get(
            f"https://api.github.com/repos/{repo}/pulls/{pr_number}/comments",
            params={"per_page": 100, "page": page},
            headers=headers,
            timeout=30,
        )
        if resp.status_code != 200:
            break
        batch = resp.json()
        if not batch:
            break
        comments.extend(batch)
        if len(batch) < 100:
            break
        page += 1

    return comments


def has_test_files(diff: str) -> bool:
    """Check if the diff touches test files."""
    test_indicators = ["test_", "_test.", ".spec.", "/test/", "/tests/", "/spec/", "test.java"]
    for line in diff.splitlines():
        if line.startswith("+++ b/") or line.startswith("--- a/"):
            file_path = line[6:]
            if any(indicator in file_path.lower() for indicator in test_indicators):
                return True
    return False


def count_diff_lines(diff: str) -> int:
    """Count total added + removed lines in a diff."""
    return sum(
        1 for line in diff.splitlines()
        if line.startswith("+") and not line.startswith("+++")
        or line.startswith("-") and not line.startswith("---")
    )


def extract_pr_outcome_pairs(
    client: httpx.Client,
    token: str,
    repo: str,
    pr_number: int,
    language: str,
) -> list[dict]:
    """
    Extract review-comment → merge-outcome pairs from a single PR.
    Returns list of training pairs (may be empty if PR doesn't meet quality filters).
    """
    # Get diff
    diff = get_pr_diff(client, token, repo, pr_number)
    if not diff:
        return []

    # Quality filters
    diff_lines = count_diff_lines(diff)
    if diff_lines > QUALITY_FILTERS["max_diff_lines"]:
        return []
    if QUALITY_FILTERS["must_have_tests"] and not has_test_files(diff):
        return []

    # Get reviews
    reviews = get_pr_reviews(client, token, repo, pr_number)
    approvals = [r for r in reviews if r.get("state") == "APPROVED"]
    if len(approvals) < QUALITY_FILTERS["min_approvals"]:
        return []

    # Get inline review comments
    review_comments = get_pr_review_comments(client, token, repo, pr_number)
    if not review_comments:
        return []

    # Build pairs
    pairs = []
    for comment in review_comments:
        comment_body = comment.get("body", "").strip()
        if len(comment_body) < 20:  # Skip trivially short comments
            continue

        file_path = comment.get("path", "")
        diff_hunk = comment.get("diff_hunk", "")

        pair = {
            "repo": repo,
            "pr_number": pr_number,
            "language": language,
            "review_comment": comment_body,
            "file_path": file_path,
            "file_context": diff_hunk,
            "diff": diff,
            "diff_lines": diff_lines,
            "has_tests": True,
            "n_approvals": len(approvals),
            "pr_merged": True,
        }
        pairs.append(pair)

    return pairs


def stream_pr_outcome_pairs(
    repos: list[dict],
    output_dir: Path,
    token: str,
    max_prs_per_repo: int = 50,
) -> Iterator[dict]:
    """
    Stream PR outcome pairs as they are discovered.
    Yields one pair at a time and writes to JSONL index.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=30) as client:
        for repo_meta in tqdm(repos, desc="Crawling repositories"):
            repo = repo_meta["full_name"]
            language = repo_meta.get("language", "Python")

            if language not in SUPPORTED_LANGUAGES:
                continue

            try:
                prs = get_merged_prs(client, token, repo, max_prs=max_prs_per_repo)
            except Exception as e:
                logger.debug(f"Failed to fetch PRs for {repo}: {e}")
                continue

            for pr in prs:
                try:
                    pairs = extract_pr_outcome_pairs(
                        client, token, repo, pr["number"], language
                    )
                    for pair in pairs:
                        yield pair
                except Exception as e:
                    logger.debug(f"Failed to extract pairs from {repo}#{pr['number']}: {e}")
                    continue

                time.sleep(0.05)  # Gentle rate limiting


def fetch_pr_outcome_pairs(
    output_dir: Path,
    n_repos: int = 50000,
    max_prs_per_repo: int = 50,
) -> None:
    """
    Main entry point. Discovers repos and collects PR outcome pairs.
    Writes JSONL to output_dir/pr_pairs_index.jsonl.
    """
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise ValueError("GITHUB_TOKEN not set. Export it: export GITHUB_TOKEN=ghp_...")

    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "pr_pairs_index.jsonl"

    logger.info(f"Fetching top {n_repos} repositories...")
    with httpx.Client(timeout=60) as client:
        repos = get_top_repos(client, token, n_repos=n_repos)

    logger.info(f"Collecting PR outcome pairs from {len(repos)} repos...")
    total = 0

    with open(index_path, "w") as f:
        for pair in stream_pr_outcome_pairs(repos, output_dir, token, max_prs_per_repo):
            f.write(json.dumps(pair) + "\n")
            total += 1
            if total % 1000 == 0:
                logger.info(f"Collected {total} PR outcome pairs so far...")

    logger.info(f"Discovery complete: {total} pairs saved to {index_path}")


if __name__ == "__main__":
    import typer

    def main(
        output_dir: str = "./data/raw/pr_pairs",
        repos: int = 50000,
        max_prs: int = 50,
        filter_only: bool = False,
        input_dir: str = "./data/raw/pr_pairs",
    ):
        if filter_only:
            # Re-filter existing pairs
            logger.info("Filter mode: re-applying quality filters to existing pairs")
            input_path = Path(input_dir) / "pr_pairs_index.jsonl"
            output_path = Path(input_dir) / "filtered_pairs.jsonl"
            if not input_path.exists():
                raise FileNotFoundError(f"Input not found: {input_path}")
            count = 0
            with open(input_path) as fin, open(output_path, "w") as fout:
                for line in fin:
                    pair = json.loads(line)
                    if (
                        pair.get("pr_merged")
                        and pair.get("has_tests")
                        and pair.get("n_approvals", 0) >= QUALITY_FILTERS["min_approvals"]
                        and pair.get("diff_lines", 9999) <= QUALITY_FILTERS["max_diff_lines"]
                    ):
                        fout.write(json.dumps(pair) + "\n")
                        count += 1
            logger.info(f"Filtered to {count} high-quality pairs")
        else:
            fetch_pr_outcome_pairs(Path(output_dir), n_repos=repos, max_prs_per_repo=max_prs)

    typer.run(main)
