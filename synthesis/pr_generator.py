"""
PR Generator — Generates improvement PRs that implement review comments

For each (original_diff, review_comment) pair:
  1. Generate the minimal code change that implements the review
  2. Validate syntax
  3. Write as a training pair
"""

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import aiofiles
import anthropic
from loguru import logger

from synthesis.prompts import PR_GENERATION_SYSTEM, PR_GENERATION_USER


@dataclass
class PRGenerationResult:
    """Result of generating an improvement PR."""

    pr_id: str
    review_comment: str
    original_diff: str
    improvement_diff: str
    commit_message: str
    syntax_valid: bool
    quality_score: float


class PRGenerator:
    """
    Generates improvement PRs that implement review suggestions.

    Target: ~30k (review, implementation) pairs.
    """

    def __init__(
        self,
        raw_dir: Path | str,
        output_dir: Path | str,
        backend: str = "claude",
        vllm_urls: Optional[list[str]] = None,
        workers: int = 20,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.vllm_urls = vllm_urls or []
        self.workers = workers
        self._semaphore = asyncio.Semaphore(workers)
        self._vllm_index = 0

        if backend == "claude":
            _api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not _api_key:
                raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set.")
            self._claude = anthropic.AsyncAnthropic(api_key=_api_key)

        self._stats = {"generated": 0, "failed": 0}

    async def generate_all(self) -> None:
        """Generate improvement PRs for all review comments in raw data."""
        prs = self._load_prs()
        logger.info(f"Generating improvement PRs for {len(prs):,} PRs...")

        output_file = self.output_dir / "pr_improvements.jsonl"
        tasks = []

        for pr in prs:
            for comment in pr.get("review_comments", [])[:3]:  # Max 3 per PR
                tasks.append(self._generate_one(pr, comment))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        async with aiofiles.open(output_file, "w") as f:
            for result in results:
                if isinstance(result, PRGenerationResult) and result.syntax_valid:
                    conversation = self._format_conversation(result)
                    await f.write(json.dumps(conversation) + "\n")
                    self._stats["generated"] += 1

        logger.info(f"PR generation: {self._stats['generated']:,} pairs generated")

    def _load_prs(self) -> list[dict]:
        """Load raw PR data."""
        prs = []
        for jsonl_file in self.raw_dir.rglob("*.jsonl"):
            with open(jsonl_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            prs.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        return prs

    async def _generate_one(
        self, pr: dict, comment: dict
    ) -> Optional[PRGenerationResult]:
        """Generate an improvement PR for one review comment."""
        async with self._semaphore:
            diff = pr.get("diff", "")
            review_body = (
                comment.get("body", "") if isinstance(comment, dict) else str(comment)
            )

            if not diff or not review_body:
                return None

            prompt = PR_GENERATION_USER.format(
                original_diff=diff[:3000],
                review_comment=review_body[:500],
            )

            response = await self._call_llm(
                PR_GENERATION_SYSTEM, prompt, max_tokens=800
            )
            if not response:
                self._stats["failed"] += 1
                return None

            # Extract diff and commit message
            improvement_diff = self._extract_diff(response)
            commit_message = self._extract_commit_message(response)
            syntax_valid = self._check_syntax(improvement_diff)

            return PRGenerationResult(
                pr_id=f"{pr.get('repo', '')}#{pr.get('pr_number', '')}",
                review_comment=review_body,
                original_diff=diff[:2000],
                improvement_diff=improvement_diff,
                commit_message=commit_message,
                syntax_valid=syntax_valid,
                quality_score=0.8 if syntax_valid else 0.2,
            )

    def _format_conversation(self, result: PRGenerationResult) -> dict:
        """Format as training conversation."""
        return {
            "conversations": [
                {
                    "role": "system",
                    "content": "You are MergePilot. Implement the code review suggestion.",
                },
                {
                    "role": "user",
                    "content": f"Original change:\n```diff\n{result.original_diff}\n```\n\nReview: {result.review_comment}",
                },
                {
                    "role": "assistant",
                    "content": f"```diff\n{result.improvement_diff}\n```\n\nCommit: {result.commit_message}",
                },
            ],
            "metadata": {
                "pr_id": result.pr_id,
                "type": "pr_improvement",
                "syntax_valid": result.syntax_valid,
            },
        }

    def _extract_diff(self, response: str) -> str:
        """Extract diff from response."""
        import re

        match = re.search(r"```diff\n(.*?)\n```", response, re.DOTALL)
        if match:
            return match.group(1)
        return ""

    def _extract_commit_message(self, response: str) -> str:
        """Extract commit message from response."""
        import re

        match = re.search(r"(?:commit|fix):\s*(.+)", response, re.IGNORECASE)
        if match:
            return match.group(1).strip()[:100]
        return "fix: implement review suggestion"

    def _check_syntax(self, diff: str) -> bool:
        """Basic syntax check — does the diff look valid?"""
        if not diff:
            return False
        lines = diff.strip().split("\n")
        has_additions = any(
            line.startswith("+") and not line.startswith("+++") for line in lines
        )
        has_context = len(lines) >= 3
        return has_additions and has_context

    async def _call_llm(self, system: str, user: str, max_tokens: int) -> Optional[str]:
        if self.backend == "claude":
            try:
                resp = await self._claude.messages.create(
                    model="claude-opus-4-6",
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                block = resp.content[0]
                return block.text if hasattr(block, "text") else None
            except Exception as e:
                logger.debug(f"Claude error: {e}")
        return None


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["claude", "vllm"], default="claude")
    parser.add_argument("--vllm-urls", nargs="+", default=[])
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()

    gen = PRGenerator(
        raw_dir="data/raw",
        output_dir="data/synthesized/pr_improvements",
        backend=args.backend,
        vllm_urls=args.vllm_urls,
        workers=args.workers,
    )
    asyncio.run(gen.generate_all())
