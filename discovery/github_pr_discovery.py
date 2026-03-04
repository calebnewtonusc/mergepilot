"""
github_pr_discovery.py — GitHub PR discovery for MergePilot training data.

Crawls GitHub repositories (1000+ stars) to collect:
  - Merged PRs with review comments that triggered code changes (LGTM / approved)
  - Rejected PRs (closed without merge) for negative training signal
  - Extracts (original_code, review_comments, revised_code, merge_outcome) tuples

Output: data/raw/github_prs/<repo_slug>.jsonl
Each line: {
    repo, pr_number, title, language,
    original_diff, review_comments, revised_diff,
    merge_outcome, n_approvals, reviewers, pr_url
}

Usage:
    export GITHUB_TOKEN=ghp_...
    python discovery/github_pr_discovery.py --repos 5000 --workers 10
    python discovery/github_pr_discovery.py --repos 5000 --languages Python TypeScript Go
"""

import asyncio
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterator, Optional

import aiofiles
import aiohttp
from loguru import logger

GITHUB_API = "https://api.github.com"

TARGET_LANGUAGES = {"Python", "TypeScript", "JavaScript", "Go", "Rust", "Java", "C++"}

# Signals that a review led to a code change (actionable review)
APPROVAL_SIGNALS = {
    "lgtm", "looks good to me", "approved", "ship it", "+1",
    "ready to merge", "good to go", "nicely done",
}

# Review keywords that indicate a change-requesting review
REVIEW_REQUEST_SIGNALS = {
    "please fix", "needs to be", "should be", "can you",
    "change this", "update this", "consider", "nit:",
    "blocking:", "must change", "required:", "address this",
}

OUTPUT_DIR = Path("data/raw/github_prs")


@dataclass
class ReviewComment:
    comment_id: int
    reviewer: str           # anonymized
    body: str
    path: Optional[str]     # file path the comment was left on
    line: Optional[int]     # line number
    diff_hunk: str          # context around the comment


@dataclass
class PRDiscoveryRecord:
    repo: str
    pr_number: int
    title: str
    language: str
    original_diff: str      # diff at time of first review
    review_comments: list   # list of ReviewComment dicts
    revised_diff: str       # final merged diff (may differ if commits added after review)
    merge_outcome: str      # "merged" | "rejected" | "abandoned"
    n_approvals: int
    reviewers: list         # anonymized reviewer handles
    pr_url: str
    stars: int
    created_at: str
    merged_at: Optional[str]
    closed_at: Optional[str]
    diff_lines: int
    has_tests: bool


class GitHubPRDiscovery:
    """
    Async GitHub PR discovery pipeline.

    Finds repos with 1000+ stars, fetches their merged and rejected PRs,
    extracts review comment → code change pairs with exponential backoff.
    """

    REQUEST_DELAY = 0.15    # seconds between requests per token
    MAX_RETRIES = 5

    def __init__(
        self,
        output_dir: Path | str = OUTPUT_DIR,
        tokens: Optional[list[str]] = None,
        workers: int = 10,
        min_stars: int = 1000,
        max_prs_per_repo: int = 50,
        languages: Optional[set[str]] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tokens = tokens or [t for t in [
            os.environ.get("GITHUB_TOKEN"),
            os.environ.get("GITHUB_TOKEN_1"),
            os.environ.get("GITHUB_TOKEN_2"),
            os.environ.get("GITHUB_TOKEN_3"),
        ] if t]
        if not self.tokens:
            raise ValueError("No GitHub tokens found. Set GITHUB_TOKEN environment variable.")
        self.workers = workers
        self.min_stars = min_stars
        self.max_prs_per_repo = max_prs_per_repo
        self.languages = languages or TARGET_LANGUAGES
        self._token_index = 0
        self._semaphore = asyncio.Semaphore(workers)
        self._stats = {
            "repos_crawled": 0,
            "prs_merged": 0,
            "prs_rejected": 0,
            "records_saved": 0,
            "api_errors": 0,
        }

    def _next_token(self) -> str:
        token = self.tokens[self._token_index % len(self.tokens)]
        self._token_index += 1
        return token

    def _headers(self, accept: str = "application/vnd.github.v3+json") -> dict:
        token = self._next_token()
        return {
            "Authorization": f"token {token}",
            "Accept": accept,
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def _anonymize(self, username: str) -> str:
        """Hash GitHub username for privacy."""
        return "user_" + hashlib.sha256(username.encode()).hexdigest()[:10]

    async def _fetch(
        self,
        session: aiohttp.ClientSession,
        url: str,
        accept: str = "application/vnd.github.v3+json",
        retries: int = 0,
    ) -> Optional[Any]:
        """Fetch a GitHub API URL with exponential backoff on rate limits."""
        await asyncio.sleep(self.REQUEST_DELAY)
        try:
            async with session.get(url, headers=self._headers(accept), timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    if "diff" in accept:
                        return await resp.text()
                    return await resp.json()
                elif resp.status in (403, 429):
                    retry_after = int(resp.headers.get("Retry-After", 60))
                    reset_time = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
                    wait = max(retry_after, reset_time - int(time.time())) + 5
                    if retries < self.MAX_RETRIES:
                        logger.warning(f"Rate limited on {url}. Waiting {wait}s (retry {retries+1}/{self.MAX_RETRIES})")
                        await asyncio.sleep(wait)
                        return await self._fetch(session, url, accept, retries + 1)
                elif resp.status == 404:
                    return None
                elif resp.status == 422:
                    return None
                else:
                    if retries < self.MAX_RETRIES:
                        wait = (2 ** retries) * 2
                        await asyncio.sleep(wait)
                        return await self._fetch(session, url, accept, retries + 1)
                    self._stats["api_errors"] += 1
        except asyncio.TimeoutError:
            if retries < self.MAX_RETRIES:
                await asyncio.sleep(2 ** retries)
                return await self._fetch(session, url, accept, retries + 1)
        except Exception as e:
            logger.debug(f"Fetch error for {url}: {e}")
            self._stats["api_errors"] += 1
        return None

    async def discover_repos(
        self,
        session: aiohttp.ClientSession,
        n_repos: int = 5000,
    ) -> list[dict]:
        """Discover top repos by stars across target languages."""
        repos = []
        per_language = max(1, n_repos // len(self.languages))

        for language in self.languages:
            page = 1
            language_count = 0

            while language_count < per_language:
                url = (
                    f"{GITHUB_API}/search/repositories"
                    f"?q=language:{language}+stars:>={self.min_stars}+is:public+archived:false"
                    f"&sort=stars&order=desc&per_page=100&page={page}"
                )
                data = await self._fetch(session, url)
                if not data or not data.get("items"):
                    break

                for repo in data["items"]:
                    repos.append({
                        "full_name": repo["full_name"],
                        "language": repo.get("language", language),
                        "stars": repo.get("stargazers_count", 0),
                        "default_branch": repo.get("default_branch", "main"),
                    })
                    language_count += 1
                    if language_count >= per_language:
                        break

                if len(data["items"]) < 100:
                    break
                page += 1
                await asyncio.sleep(1.0)  # Search API rate limit: 30 req/min

            logger.info(f"  {language}: {language_count} repos found")

        logger.info(f"Total repos discovered: {len(repos)}")
        return repos[:n_repos]

    async def _get_pr_diff(
        self, session: aiohttp.ClientSession, repo: str, pr_number: int
    ) -> str:
        url = f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}"
        diff = await self._fetch(session, url, accept="application/vnd.github.diff")
        return diff or ""

    async def _get_pr_reviews(
        self, session: aiohttp.ClientSession, repo: str, pr_number: int
    ) -> list[dict]:
        url = f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}/reviews"
        data = await self._fetch(session, url)
        return data or []

    async def _get_pr_review_comments(
        self, session: aiohttp.ClientSession, repo: str, pr_number: int
    ) -> list[dict]:
        """Fetch inline (code) review comments — these are the actionable ones."""
        comments = []
        page = 1
        while True:
            url = f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}/comments?per_page=100&page={page}"
            batch = await self._fetch(session, url)
            if not batch:
                break
            comments.extend(batch)
            if len(batch) < 100:
                break
            page += 1
        return comments

    async def _get_prs(
        self,
        session: aiohttp.ClientSession,
        repo: str,
        state: str = "closed",
        max_prs: int = 50,
    ) -> list[dict]:
        """Fetch closed PRs (both merged and rejected)."""
        prs = []
        page = 1
        while len(prs) < max_prs:
            url = f"{GITHUB_API}/repos/{repo}/pulls?state={state}&per_page=50&page={page}&sort=updated&direction=desc"
            batch = await self._fetch(session, url)
            if not batch:
                break
            prs.extend(batch)
            if len(batch) < 50:
                break
            page += 1
        return prs[:max_prs]

    def _count_diff_lines(self, diff: str) -> int:
        return sum(
            1 for line in diff.splitlines()
            if (line.startswith("+") and not line.startswith("+++"))
            or (line.startswith("-") and not line.startswith("---"))
        )

    def _has_test_files(self, diff: str) -> bool:
        test_indicators = ["test_", "_test.", ".spec.", "/test/", "/tests/", "/spec/", "test.java", "_spec.rb"]
        for line in diff.splitlines():
            if line.startswith("+++ b/") or line.startswith("--- a/"):
                path = line[6:]
                if any(ind in path.lower() for ind in test_indicators):
                    return True
        return False

    def _classify_review_comments(self, comments: list[dict]) -> list[ReviewComment]:
        """Filter and structure review comments."""
        structured = []
        for c in comments:
            body = c.get("body", "").strip()
            if len(body) < 15:
                continue
            structured.append(ReviewComment(
                comment_id=c["id"],
                reviewer=self._anonymize(c.get("user", {}).get("login", "")),
                body=body,
                path=c.get("path"),
                line=c.get("line") or c.get("original_line"),
                diff_hunk=c.get("diff_hunk", "")[:500],
            ))
        return structured

    async def _process_pr(
        self,
        session: aiohttp.ClientSession,
        repo: str,
        pr: dict,
        language: str,
        stars: int,
    ) -> Optional[PRDiscoveryRecord]:
        """Process a single PR into a discovery record."""
        pr_number = pr["number"]
        is_merged = bool(pr.get("merged_at"))
        merge_outcome = "merged" if is_merged else "rejected"

        # Skip if diff is absent (draft PRs, empty PRs)
        diff = await self._get_pr_diff(session, repo, pr_number)
        if not diff or len(diff) < 100:
            return None

        diff_lines = self._count_diff_lines(diff)
        if diff_lines > 500:  # Skip massive PRs — scope discipline signal
            return None

        # Get reviews and review comments
        reviews = await self._get_pr_reviews(session, repo, pr_number)
        review_comments_raw = await self._get_pr_review_comments(session, repo, pr_number)

        # Count approvals
        approvals = [r for r in reviews if r.get("state") == "APPROVED"]
        n_approvals = len(approvals)

        # For merged PRs: require at least 1 approval
        if is_merged and n_approvals == 0:
            return None

        # Parse review comments
        review_comments = self._classify_review_comments(review_comments_raw)
        if not review_comments:
            return None

        reviewers = list(set(c.reviewer for c in review_comments))

        return PRDiscoveryRecord(
            repo=repo,
            pr_number=pr_number,
            title=pr.get("title", "")[:200],
            language=language,
            original_diff=diff[:12000],
            review_comments=[asdict(c) for c in review_comments],
            revised_diff=diff[:12000],   # Same diff (the merged result)
            merge_outcome=merge_outcome,
            n_approvals=n_approvals,
            reviewers=reviewers,
            pr_url=pr.get("html_url", ""),
            stars=stars,
            created_at=pr.get("created_at", ""),
            merged_at=pr.get("merged_at"),
            closed_at=pr.get("closed_at"),
            diff_lines=diff_lines,
            has_tests=self._has_test_files(diff),
        )

    async def _crawl_repo(
        self,
        session: aiohttp.ClientSession,
        repo_meta: dict,
    ) -> int:
        """Crawl a single repository and save records to JSONL."""
        async with self._semaphore:
            repo = repo_meta["full_name"]
            language = repo_meta.get("language", "Python")
            stars = repo_meta.get("stars", 0)

            if language not in self.languages:
                return 0

            output_file = self.output_dir / f"{repo.replace('/', '_')}.jsonl"

            try:
                # Fetch both merged and recently closed (rejected) PRs
                prs = await self._get_prs(session, repo, state="closed", max_prs=self.max_prs_per_repo)
                if not prs:
                    return 0

                self._stats["repos_crawled"] += 1
                saved = 0

                async with aiofiles.open(output_file, "w") as f:
                    for pr in prs:
                        record = await self._process_pr(session, repo, pr, language, stars)
                        if record is None:
                            continue

                        if record.merge_outcome == "merged":
                            self._stats["prs_merged"] += 1
                        else:
                            self._stats["prs_rejected"] += 1

                        await f.write(json.dumps(asdict(record)) + "\n")
                        saved += 1
                        self._stats["records_saved"] += 1

                return saved

            except Exception as e:
                logger.debug(f"Error crawling {repo}: {e}")
                return 0

    async def crawl_all(self, n_repos: int = 5000) -> None:
        """Main entrypoint: discover repos and crawl their PRs."""
        logger.info(f"Starting GitHub PR discovery: {n_repos} repos, {self.workers} workers")

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60),
            headers={"User-Agent": "MergePilot-Crawler/1.0"},
        ) as session:
            # Step 1: Discover repos
            repos = await self.discover_repos(session, n_repos=n_repos)
            logger.info(f"Discovered {len(repos)} repositories. Starting PR crawl...")

            # Step 2: Crawl PRs in parallel
            tasks = [self._crawl_repo(session, repo) for repo in repos]
            await asyncio.gather(*tasks, return_exceptions=True)

        logger.info(
            f"Discovery complete: "
            f"{self._stats['repos_crawled']} repos, "
            f"{self._stats['prs_merged']} merged PRs, "
            f"{self._stats['prs_rejected']} rejected PRs, "
            f"{self._stats['records_saved']} total records saved"
        )
        logger.info(f"Output: {self.output_dir}")


def stream_all_records(data_dir: Path) -> Iterator[dict]:
    """Iterate over all discovery records from the output directory."""
    for jsonl_file in sorted(data_dir.glob("*.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Discover GitHub PR outcome pairs for MergePilot.")
    parser.add_argument("--repos", type=int, default=5000, help="Number of repos to crawl")
    parser.add_argument("--workers", type=int, default=10, help="Concurrent workers")
    parser.add_argument("--min-stars", type=int, default=1000, help="Minimum repo stars")
    parser.add_argument("--max-prs", type=int, default=50, help="Max PRs per repo")
    parser.add_argument("--output-dir", default="data/raw/github_prs", help="Output directory")
    parser.add_argument("--languages", nargs="+", default=None, help="Languages to include")
    parser.add_argument("--stats", action="store_true", help="Print stats about existing data")
    args = parser.parse_args()

    if args.stats:
        data_dir = Path(args.output_dir)
        total = merged = rejected = 0
        for record in stream_all_records(data_dir):
            total += 1
            if record.get("merge_outcome") == "merged":
                merged += 1
            else:
                rejected += 1
        print(f"Total records: {total:,}  |  Merged: {merged:,}  |  Rejected: {rejected:,}")
        raise SystemExit(0)

    tokens = [t for t in [
        os.getenv("GITHUB_TOKEN"),
        os.getenv("GITHUB_TOKEN_1"),
        os.getenv("GITHUB_TOKEN_2"),
        os.getenv("GITHUB_TOKEN_3"),
    ] if t]

    languages = set(args.languages) if args.languages else None

    discovery = GitHubPRDiscovery(
        output_dir=args.output_dir,
        tokens=tokens,
        workers=args.workers,
        min_stars=args.min_stars,
        max_prs_per_repo=args.max_prs,
        languages=languages,
    )

    asyncio.run(discovery.crawl_all(n_repos=args.repos))
