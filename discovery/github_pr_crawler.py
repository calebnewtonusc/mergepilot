"""
GitHub PR Crawler — Collects PR review comments from top 50k repos

Crawls:
  - GitHub REST API: PR reviews, review comments, PR diff, merge status
  - GH Archive: bulk download of PR events for historical data

Output: JSONL with diff, review comments, merge outcome per PR
"""

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Optional

import aiofiles
import aiohttp
from loguru import logger

GITHUB_API = "https://api.github.com"
GH_ARCHIVE_BASE = "https://data.gharchive.org"

# Languages to include
TARGET_LANGUAGES = {"Python", "TypeScript", "JavaScript", "Go", "Rust", "Java", "C++"}


@dataclass
class ReviewComment:
    """A single review comment on a PR."""
    comment_id: int
    reviewer: str
    body: str
    path: Optional[str]        # File being reviewed
    line: Optional[int]        # Line number
    category: Optional[str]   # Inferred review category
    actionable: bool           # Did code change in response?


@dataclass
class PullRequestData:
    """A GitHub PR with its review history and outcome."""
    repo: str
    pr_number: int
    title: str
    diff: str
    language: str
    review_comments: list[ReviewComment]
    merged: bool
    tests_pass: Optional[bool]
    merge_time_hours: Optional[float]
    author: str
    reviewers: list[str]
    pr_url: str


REVIEW_TAXONOMY = {
    "correctness_bug": ["bug", "incorrect", "wrong", "error", "off-by-one", "null pointer"],
    "security": ["injection", "xss", "auth", "csrf", "secret", "password", "sql injection"],
    "performance": ["slow", "n+1", "index", "cache", "memory leak", "inefficient", "complexity"],
    "api_design": ["naming", "interface", "signature", "api", "contract", "backwards compat"],
    "test_coverage": ["test", "coverage", "assertion", "mock", "fixture", "edge case"],
    "documentation": ["comment", "docstring", "readme", "explain", "unclear"],
    "error_handling": ["exception", "error handling", "try/catch", "failure", "graceful"],
    "type_safety": ["type", "typing", "mypy", "typescript", "cast", "any"],
    "concurrency": ["race condition", "deadlock", "thread", "async", "lock", "concurrent"],
    "code_style": ["style", "formatting", "convention", "lint", "whitespace"],
}


class GitHubPRCrawler:
    """
    Async GitHub PR crawler with rate limit handling.

    Supports multiple API tokens for higher throughput.
    """

    DELAY = 0.1  # seconds between requests (conservative)

    def __init__(
        self,
        output_dir: Path | str,
        tokens: Optional[list[str]] = None,
        workers: int = 20,
        min_stars: int = 500,
        max_prs_per_repo: int = 100,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tokens = tokens or [os.environ.get("GITHUB_TOKEN", "")]
        self.workers = workers
        self.min_stars = min_stars
        self.max_prs_per_repo = max_prs_per_repo
        self._token_index = 0
        self._semaphore = asyncio.Semaphore(workers)
        self._stats = {"repos": 0, "prs": 0, "comments": 0, "errors": 0}

    async def crawl_top_repos(self, target_count: int = 50000) -> None:
        """Crawl PRs from the top repos by stars."""
        logger.info(f"Crawling PRs from top {target_count:,} repos...")

        async with aiohttp.ClientSession(
            headers=self._headers(),
            timeout=aiohttp.ClientTimeout(total=30),
        ) as session:
            self._session = session

            # Get top repos per language
            repos = await self._get_top_repos(target_count)
            logger.info(f"Found {len(repos):,} repos to crawl")

            # Process repos in parallel
            semaphore = asyncio.Semaphore(self.workers)
            tasks = [self._crawl_repo(repo, semaphore) for repo in repos]
            await asyncio.gather(*tasks, return_exceptions=True)

        logger.info(f"Crawl complete: {self._stats}")

    async def _get_top_repos(self, count: int) -> list[dict]:
        """Get top repos by stars across target languages."""
        repos = []
        per_language = count // len(TARGET_LANGUAGES)

        for language in TARGET_LANGUAGES:
            page = 1
            while len([r for r in repos if r.get("language") == language]) < per_language:
                url = (
                    f"{GITHUB_API}/search/repositories"
                    f"?q=language:{language}+stars:>{self.min_stars}"
                    f"&sort=stars&order=desc&per_page=100&page={page}"
                )
                data = await self._fetch_json(url)
                if not data or not data.get("items"):
                    break

                for repo in data["items"]:
                    repos.append({
                        "full_name": repo["full_name"],
                        "language": repo.get("language", language),
                        "stars": repo["stargazers_count"],
                    })

                if len(data["items"]) < 100:
                    break
                page += 1

        return repos[:count]

    async def _crawl_repo(self, repo: dict, semaphore: asyncio.Semaphore) -> None:
        """Crawl PRs for a single repository."""
        async with semaphore:
            full_name = repo["full_name"]
            output_file = self.output_dir / f"{full_name.replace('/', '_')}.jsonl"

            prs = await self._get_merged_prs(full_name)
            if not prs:
                return

            self._stats["repos"] += 1

            async with aiofiles.open(output_file, "w") as f:
                for pr in prs[:self.max_prs_per_repo]:
                    data = await self._get_pr_data(full_name, pr, repo.get("language", ""))
                    if data:
                        await f.write(json.dumps(asdict(data)) + "\n")
                        self._stats["prs"] += 1
                        self._stats["comments"] += len(data.review_comments)

    async def _get_merged_prs(self, repo: str) -> list[dict]:
        """Get list of merged PRs for a repo."""
        url = f"{GITHUB_API}/repos/{repo}/pulls?state=closed&per_page=100"
        data = await self._fetch_json(url)
        if not data:
            return []
        return [pr for pr in data if pr.get("merged_at")]

    async def _get_pr_data(self, repo: str, pr: dict, language: str) -> Optional[PullRequestData]:
        """Fetch full PR data including diff and reviews."""
        pr_number = pr["number"]

        # Fetch diff
        diff_url = f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}"
        diff_data = await self._fetch_json(diff_url, accept="application/vnd.github.v3.diff")
        diff = diff_data if isinstance(diff_data, str) else ""

        if not diff or len(diff) < 50:
            return None

        # Fetch review comments
        comments_url = f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}/comments"
        comments_data = await self._fetch_json(comments_url)
        if not comments_data:
            return None

        review_comments = []
        for c in comments_data:
            body = c.get("body", "")
            if len(body) < 20:
                continue
            category = self._infer_category(body)
            review_comments.append(ReviewComment(
                comment_id=c["id"],
                reviewer=self._anonymize(c.get("user", {}).get("login", "")),
                body=body,
                path=c.get("path"),
                line=c.get("line"),
                category=category,
                actionable=False,  # Will be updated by checking subsequent commits
            ))

        if not review_comments:
            return None

        # Compute merge time
        created_at = pr.get("created_at", "")
        merged_at = pr.get("merged_at", "")
        merge_time_hours = None
        if created_at and merged_at:
            from datetime import datetime
            try:
                created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                merged = datetime.fromisoformat(merged_at.replace("Z", "+00:00"))
                merge_time_hours = (merged - created).total_seconds() / 3600
            except Exception:
                pass

        return PullRequestData(
            repo=repo,
            pr_number=pr_number,
            title=pr.get("title", ""),
            diff=diff[:8000],  # Truncate very long diffs
            language=language,
            review_comments=review_comments,
            merged=bool(pr.get("merged_at")),
            tests_pass=None,
            merge_time_hours=merge_time_hours,
            author=self._anonymize(pr.get("user", {}).get("login", "")),
            reviewers=list(set(c.reviewer for c in review_comments)),
            pr_url=pr.get("html_url", ""),
        )

    def _infer_category(self, comment: str) -> Optional[str]:
        """Infer the review category from comment text."""
        comment_lower = comment.lower()
        for category, keywords in REVIEW_TAXONOMY.items():
            if any(kw in comment_lower for kw in keywords):
                return category
        return "general"

    def _anonymize(self, username: str) -> str:
        """Anonymize GitHub username."""
        import hashlib
        return "user_" + hashlib.md5(username.encode()).hexdigest()[:8]

    def _headers(self) -> dict:
        token = self.tokens[self._token_index % len(self.tokens)]
        headers = {"Accept": "application/vnd.github.v3+json"}
        if token:
            headers["Authorization"] = f"token {token}"
        return headers

    def _rotate_token(self) -> None:
        """Advance to the next API token (call once per rate-limit event)."""
        self._token_index += 1

    async def _fetch_json(self, url: str, accept: Optional[str] = None) -> Optional[Any]:
        """Fetch JSON (or text) from GitHub API with retry and rate limit handling."""
        for attempt in range(3):
            try:
                await asyncio.sleep(self.DELAY)
                headers = self._headers()
                if accept:
                    headers["Accept"] = accept

                async with self._session.get(url, headers=headers) as resp:
                    if resp.status == 200:
                        if accept and "diff" in accept:
                            return await resp.text()
                        return await resp.json()
                    elif resp.status == 403:
                        # Rate limited — rotate to next token exactly once, then wait
                        self._rotate_token()
                        retry_after = int(resp.headers.get("Retry-After", 60))
                        logger.warning(f"Rate limited. Waiting {retry_after}s...")
                        await asyncio.sleep(retry_after)
                    elif resp.status == 404:
                        return None
                    else:
                        return None
            except Exception as e:
                if attempt == 2:
                    logger.debug(f"Failed: {url}: {e}")
                await asyncio.sleep(2 ** attempt)
        return None


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--top-repos", type=int, default=50000)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--output-dir", default="data/raw/github")
    args = parser.parse_args()

    tokens = [
        t for t in [
            os.getenv("GITHUB_TOKEN"),
            os.getenv("GITHUB_TOKEN_1"),
            os.getenv("GITHUB_TOKEN_2"),
            os.getenv("GITHUB_TOKEN_3"),
        ]
        if t
    ]

    crawler = GitHubPRCrawler(
        output_dir=args.output_dir,
        tokens=tokens,
        workers=args.workers,
    )
    asyncio.run(crawler.crawl_top_repos(args.top_repos))
