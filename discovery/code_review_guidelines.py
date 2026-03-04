"""
code_review_guidelines.py — CONTRIBUTING.md harvester for 1000+ GitHub repos.

Scrapes review standards, style guides, and merge criteria from:
  - CONTRIBUTING.md
  - docs/CONTRIBUTING.md
  - .github/CONTRIBUTING.md
  - DEVELOPMENT.md, HACKING.md, CODE_REVIEW.md

Target: 1000+ repos across Python, TypeScript, Go, Rust, Java.
Output format per record:
  {
    repo, language, stars,
    review_standards: [str],   # "All PRs require two approvals"
    style_rules: [str],        # "Use conventional commits"
    merge_criteria: [str],     # "CI must pass, changelog updated"
    testing_requirements: [str],
    raw_content: str,
    source_url: str
  }

Usage:
    export GITHUB_TOKEN=ghp_...
    python discovery/code_review_guidelines.py --count 1000
    python discovery/code_review_guidelines.py --langs python typescript
"""

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import aiofiles
import aiohttp
from loguru import logger

GITHUB_API = "https://api.github.com"
GITHUB_RAW = "https://raw.githubusercontent.com"
OUTPUT_DIR = Path(__file__).parents[1] / "data" / "raw" / "guidelines"

# File paths to check for contribution guidelines
GUIDELINE_PATHS = [
    "CONTRIBUTING.md",
    ".github/CONTRIBUTING.md",
    "docs/CONTRIBUTING.md",
    "DEVELOPMENT.md",
    "HACKING.md",
    "CODE_REVIEW.md",
    "docs/CODE_REVIEW.md",
    ".github/pull_request_template.md",
]

# High-quality repos by language with known good CONTRIBUTING guides
SEED_REPOS = {
    "python": [
        "psf/requests",
        "pallets/flask",
        "django/django",
        "fastapi/fastapi",
        "pydantic/pydantic",
        "pytest-dev/pytest",
        "encode/httpx",
        "tiangolo/sqlmodel",
        "sqlalchemy/sqlalchemy",
        "celery/celery",
        "aio-libs/aiohttp",
        "python/cpython",
        "numpy/numpy",
        "pandas-dev/pandas",
        "scikit-learn/scikit-learn",
        "torvalds/linux",
        "pypa/pip",
        "python-poetry/poetry",
    ],
    "typescript": [
        "microsoft/TypeScript",
        "microsoft/vscode",
        "vercel/next.js",
        "facebook/react",
        "angular/angular",
        "vuejs/vue",
        "sveltejs/svelte",
        "denoland/deno",
        "nicolo-ribaudo/babel",
        "prisma/prisma",
        "trpc/trpc",
        "colinhacks/zod",
        "supabase/supabase",
        "shadcn-ui/ui",
        "tailwindlabs/tailwindcss",
    ],
    "go": [
        "golang/go",
        "kubernetes/kubernetes",
        "hashicorp/terraform",
        "docker/docker",
        "gin-gonic/gin",
        "go-chi/chi",
        "prometheus/prometheus",
        "grafana/grafana",
        "etcd-io/etcd",
        "helm/helm",
        "cilium/cilium",
        "spf13/cobra",
    ],
    "rust": [
        "rust-lang/rust",
        "tokio-rs/tokio",
        "actix/actix-web",
        "serde-rs/serde",
        "clap-rs/clap",
        "BurntSushi/ripgrep",
        "sharkdp/fd",
        "starship/starship",
        "alacritty/alacritty",
    ],
    "java": [
        "spring-projects/spring-framework",
        "apache/kafka",
        "elastic/elasticsearch",
        "square/okhttp",
        "google/guava",
        "ReactiveX/RxJava",
        "junit-team/junit5",
        "hibernate/hibernate-orm",
    ],
}

# Review-related keywords for extraction
REVIEW_KEYWORDS = re.compile(
    r"\b(review|approve|lgtm|merge|pr|pull request|code review|"
    r"reviewer|maintainer|contributor|codeowner)\b",
    re.IGNORECASE,
)
STYLE_KEYWORDS = re.compile(
    r"\b(style|format|lint|linter|prettier|black|eslint|gofmt|rustfmt|"
    r"conventional commit|commit message|naming convention)\b",
    re.IGNORECASE,
)
TESTING_KEYWORDS = re.compile(
    r"\b(test|tests|testing|coverage|unit test|integration test|"
    r"e2e|benchmark|ci|ci\/cd|github actions|passing)\b",
    re.IGNORECASE,
)
MERGE_KEYWORDS = re.compile(
    r"\b(merge|squash|rebase|fast.forward|approval|approvals|"
    r"changelog|release note|breaking change|semver)\b",
    re.IGNORECASE,
)


@dataclass
class ReviewGuidelines:
    """Parsed review guidelines from a single repository."""

    repo: str
    language: str
    stars: int
    review_standards: list[str]
    style_rules: list[str]
    merge_criteria: list[str]
    testing_requirements: list[str]
    raw_content: str
    source_url: str
    source_file: str


def _extract_bullet_points(content: str) -> list[str]:
    """Extract bullet points and numbered list items from markdown."""
    bullets = []
    for line in content.splitlines():
        stripped = line.strip()
        # Match - item, * item, + item, 1. item
        m = re.match(r"^[-*+]\s+(.+)$", stripped)
        if not m:
            m = re.match(r"^\d+\.\s+(.+)$", stripped)
        if m:
            text = m.group(1).strip()
            # Filter out very short or very long items
            if 20 < len(text) < 300:
                bullets.append(text)
    return bullets


def _extract_section(content: str, section_regex: re.Pattern) -> list[str]:
    """Extract lines from a section matching the regex."""
    lines_in_section = []
    in_section = False

    for line in content.splitlines():
        # Check if this line starts a matching section heading
        if re.search(r"^#{1,3}\s+", line):
            in_section = bool(section_regex.search(line))
            continue

        if in_section:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                if section_regex.search(stripped) or len(stripped) > 30:
                    lines_in_section.append(stripped)
            elif stripped.startswith("#"):
                in_section = False

    return lines_in_section[:20]  # Cap at 20 items per section


def parse_guidelines(
    content: str,
    repo: str,
    language: str,
    stars: int,
    source_url: str,
    source_file: str,
) -> ReviewGuidelines:
    """Parse a CONTRIBUTING.md into structured review guidelines."""
    all_bullets = _extract_bullet_points(content)

    # Classify bullets by keyword
    review_standards = [b for b in all_bullets if REVIEW_KEYWORDS.search(b)]
    style_rules = [b for b in all_bullets if STYLE_KEYWORDS.search(b)]
    merge_criteria = [b for b in all_bullets if MERGE_KEYWORDS.search(b)]
    testing_requirements = [b for b in all_bullets if TESTING_KEYWORDS.search(b)]

    # Fallback: extract sentences containing keywords
    if not review_standards:
        sentences = re.split(r"[.!?]\s+", content)
        review_standards = [
            s.strip()
            for s in sentences
            if REVIEW_KEYWORDS.search(s) and 30 < len(s.strip()) < 300
        ][:10]

    if not testing_requirements:
        sentences = re.split(r"[.!?]\s+", content)
        testing_requirements = [
            s.strip()
            for s in sentences
            if TESTING_KEYWORDS.search(s) and 30 < len(s.strip()) < 300
        ][:10]

    return ReviewGuidelines(
        repo=repo,
        language=language,
        stars=stars,
        review_standards=review_standards[:15],
        style_rules=style_rules[:15],
        merge_criteria=merge_criteria[:15],
        testing_requirements=testing_requirements[:15],
        raw_content=content[:8000],
        source_url=source_url,
        source_file=source_file,
    )


class GuidelinesHarvester:
    """
    Async harvester for CONTRIBUTING.md files from GitHub.

    Two modes:
      1. Seed repos: Fetch from predefined high-quality repositories
      2. Search mode: Use GitHub code search to find CONTRIBUTING.md files
    """

    REQUEST_DELAY = 0.15

    def __init__(
        self,
        output_dir: Path = OUTPUT_DIR,
        token: Optional[str] = None,
        workers: int = 20,
        languages: Optional[list[str]] = None,
        target_count: int = 1000,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.token = token or os.environ.get("GITHUB_TOKEN", "")
        self.workers = workers
        self.languages = languages or list(SEED_REPOS.keys())
        self.target_count = target_count
        self._semaphore = asyncio.Semaphore(workers)
        self._stats = {"fetched": 0, "parsed": 0, "errors": 0}

    def _headers(self) -> dict:
        h = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "MergePilot-Research/1.0",
        }
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    async def _fetch_json(
        self, session: aiohttp.ClientSession, url: str, params: Optional[dict] = None
    ) -> Optional[dict]:
        """Fetch GitHub API JSON with rate limit handling."""
        for attempt in range(4):
            await asyncio.sleep(self.REQUEST_DELAY)
            try:
                async with session.get(
                    url,
                    headers=self._headers(),
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    elif resp.status in (403, 429):
                        remaining = int(resp.headers.get("X-RateLimit-Remaining", 1))
                        if remaining < 5:
                            reset = int(
                                resp.headers.get("X-RateLimit-Reset", time.time() + 60)
                            )
                            wait = max(1, reset - int(time.time())) + 5
                            logger.warning(f"Rate limited. Waiting {wait}s...")
                            await asyncio.sleep(wait)
                        else:
                            await asyncio.sleep(2**attempt)
                    elif resp.status == 404:
                        return None
                    else:
                        await asyncio.sleep(2**attempt)
            except Exception as e:
                logger.debug(f"Fetch error {url}: {e}")
                await asyncio.sleep(2**attempt)
        return None

    async def _fetch_raw(
        self, session: aiohttp.ClientSession, url: str
    ) -> Optional[str]:
        """Fetch raw file content."""
        for attempt in range(3):
            await asyncio.sleep(0.05)
            try:
                async with session.get(
                    url,
                    headers={"User-Agent": "MergePilot-Research/1.0"},
                    timeout=aiohttp.ClientTimeout(total=20),
                ) as resp:
                    if resp.status == 200:
                        return await resp.text(errors="replace")
                    elif resp.status == 429:
                        await asyncio.sleep(5 * (attempt + 1))
                    else:
                        return None
            except Exception as e:
                logger.debug(f"Raw fetch error: {e}")
                await asyncio.sleep(2**attempt)
        return None

    async def _get_repo_info(
        self, session: aiohttp.ClientSession, repo: str
    ) -> Optional[dict]:
        """Get repository metadata (stars, default branch, language)."""
        url = f"{GITHUB_API}/repos/{repo}"
        return await self._fetch_json(session, url)

    async def _fetch_guideline_file(
        self,
        session: aiohttp.ClientSession,
        owner: str,
        repo: str,
        branch: str,
        path: str,
    ) -> Optional[str]:
        """Try to fetch a specific guideline file from a repo."""
        url = f"{GITHUB_RAW}/{owner}/{repo}/{branch}/{path}"
        return await self._fetch_raw(session, url)

    async def _process_repo(
        self,
        session: aiohttp.ClientSession,
        repo_slug: str,
        language: str,
        output_file: Path,
    ) -> bool:
        """Fetch and parse guidelines for a single repo."""
        async with self._semaphore:
            owner, repo = repo_slug.split("/", 1)

            repo_info = await self._get_repo_info(session, repo_slug)
            if not repo_info:
                return False

            stars = repo_info.get("stargazers_count", 0)
            branch = repo_info.get("default_branch", "main")
            detected_language = repo_info.get("language", language) or language

            # Try each guideline file path
            for path in GUIDELINE_PATHS:
                content = await self._fetch_guideline_file(
                    session, owner, repo, branch, path
                )
                if content and len(content) > 200:
                    source_url = f"https://github.com/{repo_slug}/blob/{branch}/{path}"
                    guidelines = parse_guidelines(
                        content,
                        repo=repo_slug,
                        language=detected_language.lower(),
                        stars=stars,
                        source_url=source_url,
                        source_file=path,
                    )

                    if guidelines.review_standards or guidelines.testing_requirements:
                        async with aiofiles.open(output_file, "a") as f:
                            await f.write(json.dumps(asdict(guidelines)) + "\n")
                        self._stats["parsed"] += 1
                        return True

            self._stats["errors"] += 1
            return False

    async def _search_repos_by_language(
        self,
        session: aiohttp.ClientSession,
        language: str,
        count: int,
    ) -> list[str]:
        """
        Search GitHub for popular repos in a language that have CONTRIBUTING.md.

        Uses GitHub search API: stars:>100 language:{lang} sort:stars
        """
        repos = []
        page = 1

        while len(repos) < count:
            params = {
                "q": f"language:{language} stars:>500 has:contributing",
                "sort": "stars",
                "order": "desc",
                "per_page": 100,
                "page": page,
            }
            data = await self._fetch_json(
                session, f"{GITHUB_API}/search/repositories", params=params
            )

            if not data or "items" not in data:
                break

            items = data.get("items", [])
            if not items:
                break

            for item in items:
                full_name = item.get("full_name", "")
                if full_name:
                    repos.append(full_name)
                if len(repos) >= count:
                    break

            page += 1
            # Respect GitHub search rate limit (10 req/min for unauthenticated)
            await asyncio.sleep(1.0)

        return repos[:count]

    async def harvest_all(self) -> int:
        """Harvest guidelines from all configured languages."""
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=50, limit_per_host=10),
        ) as session:
            all_tasks = []

            for language in self.languages:
                # Start with seed repos
                seed_repos = SEED_REPOS.get(language, [])
                output_file = self.output_dir / f"{language}.jsonl"

                # Add search repos to fill up to target
                remaining = max(
                    0, self.target_count // len(self.languages) - len(seed_repos)
                )
                search_repos = []
                if remaining > 0:
                    logger.info(
                        f"Searching {remaining} more repos for language: {language}"
                    )
                    search_repos = await self._search_repos_by_language(
                        session, language, remaining
                    )

                repos = list(
                    dict.fromkeys(seed_repos + search_repos)
                )  # deduplicate, preserve order
                logger.info(f"Processing {len(repos)} repos for {language}")

                for repo in repos:
                    self._stats["fetched"] += 1
                    all_tasks.append(
                        self._process_repo(session, repo, language, output_file)
                    )

            results = await asyncio.gather(*all_tasks, return_exceptions=True)
            success_count = sum(1 for r in results if r is True)

        logger.success(
            f"Guidelines harvest complete: "
            f"{self._stats['parsed']} parsed, "
            f"{self._stats['errors']} errors"
        )
        return success_count


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Harvest GitHub CONTRIBUTING.md guidelines"
    )
    parser.add_argument(
        "--count", type=int, default=1000, help="Target number of repos"
    )
    parser.add_argument(
        "--langs",
        nargs="+",
        default=None,
        choices=list(SEED_REPOS.keys()),
        help="Languages to harvest (default: all)",
    )
    parser.add_argument("--output-dir", default="data/raw/guidelines")
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()

    harvester = GuidelinesHarvester(
        output_dir=args.output_dir,
        workers=args.workers,
        languages=args.langs,
        target_count=args.count,
    )
    n = asyncio.run(harvester.harvest_all())
    print(f"\nTotal guidelines harvested: {n:,}")
