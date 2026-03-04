"""
MergePilot Synthesis Pipeline — Bulk async synthesis

Generates structured review → implementation pairs from raw PR data.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Optional

import aiofiles
import aiohttp
import anthropic
from loguru import logger

from synthesis.prompts import REVIEW_SYSTEM, REVIEW_USER


@dataclass
class SynthesisResult:
    """Result of synthesizing one review pair."""

    pr_id: str
    conversation: dict
    quality_score: float
    success: bool
    error: Optional[str] = None


class SynthesisPipeline:
    """Async pipeline for bulk synthesis of code review interactions."""

    def __init__(
        self,
        raw_dir: Path | str,
        output_dir: Path | str,
        backend: str = "claude",
        vllm_urls: Optional[list[str]] = None,
        vllm_model: str = "Qwen/Qwen2.5-7B-Coder-Instruct",
        workers: int = 20,
        min_quality_score: float = 0.6,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.vllm_urls = vllm_urls or []
        self.vllm_model = vllm_model or os.getenv(
            "MODEL_PATH", "Qwen/Qwen2.5-7B-Coder-Instruct"
        )
        self.workers = workers
        self.min_quality_score = min_quality_score
        self._semaphore = asyncio.Semaphore(workers)
        self._vllm_index = 0

        if backend == "claude":
            _api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not _api_key:
                raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set.")
            self._claude = anthropic.AsyncAnthropic(api_key=_api_key)

        self._stats = {"processed": 0, "success": 0, "failed": 0}

    async def synthesize_all(self) -> int:
        """Synthesize review pairs for all PRs in raw_dir. Returns success count."""
        prs = self._load_all_prs()
        logger.info(f"Synthesizing reviews for {len(prs):,} PRs...")

        tasks = [self._synthesize_one(pr) for pr in prs]
        await asyncio.gather(*tasks, return_exceptions=True)

        logger.info(
            f"Synthesis: {self._stats['success']:,} success, "
            f"{self._stats['failed']:,} failed"
        )
        return self._stats["success"]

    def _load_all_prs(self) -> list[dict]:
        """Load all PRs from raw data."""
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

    async def _synthesize_one(self, pr: dict) -> Optional[SynthesisResult]:
        """Synthesize a review for a single PR."""
        async with self._semaphore:
            diff = pr.get("diff", "")
            if not diff or len(diff) < 100:
                return None

            prompt = REVIEW_USER.format(
                repo=pr.get("repo", "unknown"),
                language=pr.get("language", "Python"),
                title=pr.get("title", ""),
                diff=diff[:4000],
                context="",
            )

            response = await self._call_llm(REVIEW_SYSTEM, prompt, max_tokens=1000)
            if not response:
                self._stats["failed"] += 1
                return None

            quality = self._score_quality(response)
            if quality < self.min_quality_score:
                return None

            conversation = {
                "conversations": [
                    {
                        "role": "system",
                        "content": "You are MergePilot, an expert code reviewer.",
                    },
                    {
                        "role": "user",
                        "content": f"PR diff:\n```diff\n{diff[:3000]}\n```",
                    },
                    {"role": "assistant", "content": response},
                ],
                "metadata": {
                    "repo": pr.get("repo", ""),
                    "pr_number": pr.get("pr_number"),
                    "merged": pr.get("merged", False),
                    "language": pr.get("language", ""),
                    "quality_score": quality,
                },
            }

            output_file = (
                self.output_dir / f"{pr.get('language', 'unknown').lower()}.jsonl"
            )
            async with aiofiles.open(output_file, "a") as f:
                await f.write(json.dumps(conversation) + "\n")

            self._stats["success"] += 1
            return SynthesisResult(
                pr_id=f"{pr.get('repo', '')}#{pr.get('pr_number', '')}",
                conversation=conversation,
                quality_score=quality,
                success=True,
            )

    def _score_quality(self, response: str) -> float:
        """Score review quality."""
        score = 0.5
        response_lower = response.lower()

        # Has category tags
        categories = [
            "[correctness]",
            "[security]",
            "[performance]",
            "[tests]",
            "[docs]",
        ]
        score += 0.1 * sum(1 for c in categories if c in response_lower)

        # Has file/line references
        import re

        if re.search(r"file:|line:|path:", response_lower):
            score += 0.1

        # Has code examples
        if "```" in response:
            score += 0.1

        # Reasonable length
        if 200 < len(response) < 3000:
            score += 0.1

        return min(1.0, score)

    async def _call_llm(self, system: str, user: str, max_tokens: int) -> Optional[str]:
        if self.backend == "claude":
            return await self._call_claude(system, user, max_tokens)
        return await self._call_vllm(system, user, max_tokens)

    async def _call_claude(
        self, system: str, user: str, max_tokens: int
    ) -> Optional[str]:
        try:
            resp = await self._claude.messages.create(
                model="claude-opus-4-6",
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return resp.content[0].text
        except Exception as e:
            logger.debug(f"Claude error: {e}")
            return None

    async def _call_vllm(
        self, system: str, user: str, max_tokens: int
    ) -> Optional[str]:
        if not self.vllm_urls:
            return None
        url = self.vllm_urls[self._vllm_index % len(self.vllm_urls)]
        self._vllm_index += 1
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{url}/v1/chat/completions",
                    json={
                        "model": self.vllm_model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        "max_tokens": max_tokens,
                        "temperature": 0.7,
                    },
                    headers={
                        "Authorization": f"Bearer {os.getenv('VLLM_API_KEY', '')}"
                    },
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.debug(f"vLLM error: {e}")
        return None


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Bulk synthesis of code review training data"
    )
    parser.add_argument("--backend", choices=["claude", "vllm"], default="claude")
    parser.add_argument("--vllm-urls", nargs="+", default=[])
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--output-dir", default="data/synthesized")
    parser.add_argument("--raw-dir", default="data/raw")
    args = parser.parse_args()

    pipeline = SynthesisPipeline(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        backend=args.backend,
        vllm_urls=args.vllm_urls,
        workers=args.workers,
    )
    asyncio.run(pipeline.synthesize_all())
