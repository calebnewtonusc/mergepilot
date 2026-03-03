"""
pr_improver.py — Generate improved code diffs from (diff + review_comments) inputs.

Given a raw PR diff and review comments, produces:
  1. minimal_diff: the smallest change that addresses all review comments
  2. tests: test code that verifies the fix
  3. reasoning: explanation of what changed and why

Supports both Claude and vLLM backends.

Usage:
    python synthesis/pr_improver.py \\
        --input data/raw/github_prs/python.jsonl \\
        --output data/synthesized/improved_prs.jsonl \\
        --backend claude

    python synthesis/pr_improver.py \\
        --input data/raw/github_prs/typescript.jsonl \\
        --output data/synthesized/improved_prs.jsonl \\
        --backend vllm \\
        --vllm-urls http://localhost:8001/v1 http://localhost:8002/v1
"""

import asyncio
import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import aiofiles
import aiohttp
import anthropic
from loguru import logger

OUTPUT_DIR = Path(__file__).parents[1] / "data" / "synthesized"

# ── System Prompt ─────────────────────────────────────────────────────────────

PR_IMPROVER_SYSTEM = """\
You are MergePilot, an expert code reviewer and pull request author.

Given a PR diff and reviewer comments, you produce:
1. A MINIMAL diff that addresses ONLY what was requested in the review comments
2. Test code proving the fix works
3. Clear reasoning connecting the review comment to the change

Rules for minimal diffs:
- Change ONLY the lines needed to address the specific comment
- Do NOT refactor, rename, or reorganize unrelated code
- Do NOT add features beyond what was asked
- Do NOT change whitespace/formatting unless that was the review request
- Each change must have a clear mapping to a review comment

Output format (JSON):
{
  "review_addressed": "one-sentence summary of what was fixed",
  "minimal_diff": "unified diff format (--- a/file.py, +++ b/file.py, @@ ... @@)",
  "reasoning": "why this specific change addresses the comment",
  "tests": "test code that verifies the fix (pytest/jest/go test style)",
  "scope_discipline": "what makes this change appropriately scoped"
}
"""

PR_IMPROVER_USER_TEMPLATE = """\
Repository: {repo}
Language: {language}
PR Title: {title}

Review comment (what the reviewer asked for):
{review_comment}

Code context where the comment was left:
```{language}
{file_context}
```

Original PR diff (what was actually merged):
```diff
{diff}
```

Generate the MINIMAL diff that addresses this specific review comment, plus tests.
Return ONLY valid JSON matching the specified format.
"""

# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class ImprovedPR:
    """A PR improved by addressing review comments."""
    repo: str
    pr_number: Optional[int]
    language: str
    title: str
    review_comment: str
    original_diff: str
    minimal_diff: str
    tests: str
    reasoning: str
    scope_discipline: str
    review_addressed: str
    quality_score: float
    backend: str


# ── Quality Scoring ───────────────────────────────────────────────────────────

def score_improvement(result: dict, review_comment: str) -> float:
    """Score the quality of a PR improvement (0.0 – 1.0)."""
    score = 0.3

    minimal_diff = result.get("minimal_diff", "")
    tests = result.get("tests", "")
    reasoning = result.get("reasoning", "")

    # Has actual diff content
    if minimal_diff and ("@@" in minimal_diff or "+" in minimal_diff or "-" in minimal_diff):
        score += 0.2

    # Has test code
    if tests and len(tests) > 50:
        score += 0.2

    # Has reasoning that references the comment
    if reasoning and len(reasoning) > 50:
        score += 0.1

    # Diff is shorter than 200 lines (minimal principle)
    diff_lines = minimal_diff.count("\n")
    if 2 <= diff_lines <= 50:
        score += 0.1
    elif diff_lines <= 200:
        score += 0.05

    # Review keywords appear in reasoning
    comment_words = set(review_comment.lower().split())
    reasoning_words = set(reasoning.lower().split())
    overlap = len(comment_words & reasoning_words)
    if overlap >= 3:
        score += 0.1

    return round(min(1.0, score), 3)


# ── LLM Interface ─────────────────────────────────────────────────────────────

class PRImprover:
    """
    Generates improved PR diffs from review comments.

    Supports both Claude and vLLM (OpenAI-compatible) backends.
    """

    def __init__(
        self,
        input_path: Path,
        output_path: Path,
        backend: str = "claude",
        vllm_urls: Optional[list[str]] = None,
        vllm_model: str = "Qwen/Qwen2.5-72B-Instruct",
        workers: int = 20,
        min_quality_score: float = 0.5,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> None:
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.vllm_urls = vllm_urls or []
        self.vllm_model = vllm_model
        self.workers = workers
        self.min_quality_score = min_quality_score
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._semaphore = asyncio.Semaphore(workers)
        self._vllm_index = 0
        self._stats = {"total": 0, "success": 0, "failed": 0, "filtered": 0}

        if backend == "claude":
            self._claude = anthropic.AsyncAnthropic(
                api_key=os.environ["ANTHROPIC_API_KEY"]
            )

    async def _call_claude(self, user_prompt: str) -> Optional[str]:
        """Call Claude API."""
        try:
            resp = await self._claude.messages.create(
                model="claude-opus-4-6",
                max_tokens=self.max_tokens,
                system=PR_IMPROVER_SYSTEM,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=self.temperature,
            )
            return resp.content[0].text
        except Exception as e:
            logger.debug(f"Claude error: {e}")
            return None

    async def _call_vllm(self, user_prompt: str) -> Optional[str]:
        """Call vLLM server (round-robin across instances)."""
        if not self.vllm_urls:
            logger.error("No vLLM URLs configured")
            return None

        url = self.vllm_urls[self._vllm_index % len(self.vllm_urls)]
        self._vllm_index += 1

        payload = {
            "model": self.vllm_model,
            "messages": [
                {"role": "system", "content": PR_IMPROVER_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {os.getenv('VLLM_API_KEY', 'synthesis')}",
            "Content-Type": "application/json",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data["choices"][0]["message"]["content"]
                    else:
                        logger.debug(f"vLLM {resp.status}: {await resp.text()}")
                        return None
        except Exception as e:
            logger.debug(f"vLLM error: {e}")
            return None

    async def _call_llm(self, user_prompt: str) -> Optional[str]:
        """Route to configured backend."""
        if self.backend == "claude":
            return await self._call_claude(user_prompt)
        return await self._call_vllm(user_prompt)

    def _extract_json(self, response: str) -> Optional[dict]:
        """Extract JSON from LLM response."""
        if not response:
            return None

        # Try direct parse first
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Extract from code block
        code_match = re.search(r'```(?:json)?\s*\n(.*?)```', response, re.DOTALL)
        if code_match:
            try:
                return json.loads(code_match.group(1))
            except json.JSONDecodeError:
                pass

        # Extract object from anywhere in the response
        obj_match = re.search(r'\{[^{}]*"minimal_diff"[^{}]*\}', response, re.DOTALL)
        if obj_match:
            try:
                return json.loads(obj_match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    async def _improve_one(self, pr: dict) -> Optional[ImprovedPR]:
        """Improve a single PR by addressing its review comments."""
        async with self._semaphore:
            review_comment = pr.get("review_comment", "")
            diff = pr.get("diff", "")

            if not review_comment or not diff:
                return None

            # Build prompt
            user_prompt = PR_IMPROVER_USER_TEMPLATE.format(
                repo=pr.get("repo", "unknown/repo"),
                language=pr.get("language", "python"),
                title=pr.get("title", ""),
                review_comment=review_comment[:1000],
                file_context=pr.get("file_context", "")[:3000],
                diff=diff[:4000],
            )

            response = await self._call_llm(user_prompt)
            if not response:
                self._stats["failed"] += 1
                return None

            result = self._extract_json(response)
            if not result:
                self._stats["failed"] += 1
                return None

            quality = score_improvement(result, review_comment)
            self._stats["total"] += 1

            if quality < self.min_quality_score:
                self._stats["filtered"] += 1
                return None

            self._stats["success"] += 1

            return ImprovedPR(
                repo=pr.get("repo", ""),
                pr_number=pr.get("pr_number"),
                language=pr.get("language", "python"),
                title=pr.get("title", ""),
                review_comment=review_comment,
                original_diff=diff[:4000],
                minimal_diff=result.get("minimal_diff", ""),
                tests=result.get("tests", ""),
                reasoning=result.get("reasoning", ""),
                scope_discipline=result.get("scope_discipline", ""),
                review_addressed=result.get("review_addressed", ""),
                quality_score=quality,
                backend=self.backend,
            )

    def _load_prs(self) -> list[dict]:
        """Load PR records from JSONL."""
        records = []
        with open(self.input_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    async def improve_all(self) -> int:
        """Improve all PRs and write results."""
        logger.info(f"Loading PRs from {self.input_path}")
        prs = self._load_prs()
        logger.info(f"Processing {len(prs):,} PRs with {self.backend} backend...")

        tasks = [self._improve_one(pr) for pr in prs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid = [r for r in results if isinstance(r, ImprovedPR)]
        logger.info(f"Writing {len(valid):,} improved PRs to {self.output_path}")

        async with aiofiles.open(self.output_path, "w") as f:
            for result in valid:
                await f.write(json.dumps(asdict(result)) + "\n")

        logger.success(
            f"PR improvement complete: "
            f"{self._stats['success']} success, "
            f"{self._stats['failed']} failed, "
            f"{self._stats['filtered']} filtered (score < {self.min_quality_score})"
        )
        return len(valid)


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Generate improved PR diffs from review comments")
    parser.add_argument("--input", required=True, help="Input JSONL with PR records")
    parser.add_argument("--output", default="data/synthesized/improved_prs.jsonl")
    parser.add_argument("--backend", choices=["claude", "vllm"], default="claude")
    parser.add_argument("--vllm-urls", nargs="+", default=None,
                        help="vLLM server URLs (e.g. http://localhost:8001/v1)")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--min-score", type=float, default=0.5)
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()

    improver = PRImprover(
        input_path=Path(args.input),
        output_path=Path(args.output),
        backend=args.backend,
        vllm_urls=args.vllm_urls,
        workers=args.workers,
        min_quality_score=args.min_score,
        max_tokens=args.max_tokens,
    )
    n = asyncio.run(improver.improve_all())
    print(f"\nTotal improved PRs: {n:,}")
