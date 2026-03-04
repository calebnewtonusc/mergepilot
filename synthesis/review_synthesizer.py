"""
review_synthesizer.py — Persona-based code review synthesizer.

Generates structured review comments from PR diffs using multiple reviewer personas:
  - Security Reviewer: focuses on auth, injection, SSRF, secrets exposure
  - Performance Reviewer: allocations, DB queries, caching, algorithmic complexity
  - Correctness Reviewer: logic errors, edge cases, off-by-ones, null handling
  - Testing Reviewer: missing tests, coverage gaps, flaky tests
  - Documentation Reviewer: missing docstrings, unclear variable names, API changes
  - Maintainability Reviewer: code duplication, coupling, SOLID principles

Supports both Claude and vLLM backends.
Generates both SFT pairs (review comment → minimal diff) and DPO pairs
(minimal vs. bloated diff).

Usage:
    python synthesis/review_synthesizer.py \\
        --input data/raw/github_prs/python.jsonl \\
        --output data/synthesized/review_pairs.jsonl \\
        --backend claude --personas security performance correctness

    python synthesis/review_synthesizer.py \\
        --backend vllm \\
        --vllm-urls http://localhost:8001/v1 http://localhost:8002/v1
"""

import asyncio
import itertools
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

# ── Reviewer Personas ─────────────────────────────────────────────────────────

REVIEWER_PERSONAS = {
    "security": """\
You are a security-focused code reviewer with expertise in OWASP Top 10, secure coding practices,
and common vulnerability patterns. You notice:
- Authentication and authorization bypasses
- SQL injection, command injection, path traversal
- SSRF, CSRF, XSS vulnerabilities
- Hardcoded secrets, insecure randomness
- Missing input validation, unsafe deserialization
- Overly permissive CORS, weak TLS configuration

Your reviews are specific: you cite the exact line and mechanism of the vulnerability.
Never raise false positives — if something COULD be a problem but is demonstrably safe, note that.""",

    "performance": """\
You are a performance-focused code reviewer who understands memory layout, CPU caches,
database query plans, and async I/O. You notice:
- N+1 query problems (DB queries in loops)
- Missing database indexes for common access patterns
- Unnecessary allocations in hot paths
- Synchronous blocking calls in async contexts
- Inefficient data structures (O(n) lookups where O(1) exists)
- Missing caching for expensive computations
- Unbounded memory growth (appending to list in loop without bound)

You always quantify: "this will make 10 extra DB queries per request" not just "N+1 query".""",

    "correctness": """\
You are a correctness-focused code reviewer who catches logical errors, edge cases, and subtle bugs.
You notice:
- Off-by-one errors in loops and array indexing
- Missing null/None checks before dereference
- Integer overflow/underflow
- Race conditions in concurrent code
- Wrong comparison operators (= vs ==, is vs ==)
- Logic inversions (if x vs if not x)
- Missing error handling for exceptional cases
- Incorrect use of floating point equality

You explain WHY the code is wrong with a concrete counterexample.""",

    "testing": """\
You are a testing-focused code reviewer who ensures code is properly verified.
You notice:
- Missing tests for new public functions
- Untested error paths and edge cases
- Tests that only test the happy path
- Mocking at the wrong level (testing mocks, not behavior)
- Missing integration tests for external dependencies
- Tests that are too brittle (testing implementation, not interface)
- Missing property-based tests for algorithmic code

You suggest specific test cases: "test with empty list, None, and list of one element".""",

    "documentation": """\
You are a documentation-focused code reviewer ensuring code is understandable.
You notice:
- Missing docstrings for public functions/classes/modules
- Parameters and return types not documented
- Missing examples for complex functions
- Outdated documentation (says X does Y, but code does Z)
- Unclear variable names (x, temp, data)
- Missing type annotations for Python code
- API changes without updating changelog or README

You suggest specific improvements: show the exact docstring that should be added.""",

    "maintainability": """\
You are a maintainability-focused code reviewer ensuring code is clean and extensible.
You notice:
- Code duplication (copy-paste code that should be extracted)
- Functions longer than 50 lines that should be split
- High coupling between unrelated components
- Missing abstractions (concrete classes where interfaces would be cleaner)
- Magic numbers and magic strings
- Global mutable state
- Inconsistent naming conventions within the same file

You suggest refactors that reduce complexity without changing behavior.""",
}

# ── Synthesis Prompts ─────────────────────────────────────────────────────────

REVIEW_SYNTHESIS_USER_TEMPLATE = """\
Repository: {repo}
Language: {language}
PR Title: {title}

PR Diff:
```diff
{diff}
```

{persona_context}

Review this PR diff from your perspective. Generate a structured code review comment.

Your review must:
1. Be actionable: specify exactly what to change and why
2. Reference specific file paths, function names, or line content
3. Explain the impact if NOT fixed (security risk, performance regression, bug)
4. Be proportional: blocking issues vs. advisory suggestions

Format as JSON:
{{
  "review_type": "blocking|advisory|style|security|performance|test",
  "file_path": "the file this comment is on (if identifiable from diff)",
  "comment": "the full review comment as you would write it on GitHub",
  "impact": "what happens if this is not addressed",
  "minimal_fix": "the exact code change that would address this comment (unified diff or code snippet)",
  "test_for_fix": "a test that verifies the fix works correctly"
}}

Return ONLY valid JSON.
"""

DPO_GENERATION_USER_TEMPLATE = """\
A code reviewer left this comment:
"{review_comment}"

The minimal correct fix is:
```
{minimal_fix}
```

Now generate a "bloated" version of this fix that technically addresses the comment
but includes unnecessary extra changes:
- Extra refactoring not requested
- Rewriting related code that works fine
- Adding abstractions not needed here
- Extra logging/comments not requested

Return JSON with exactly this structure (copy the minimal fix above verbatim into "chosen"):
{{
  "chosen": <copy the minimal fix shown above verbatim>,
  "rejected": "...(the bloated version with unnecessary changes)...",
  "chosen_explanation": "why the minimal fix is better",
  "rejected_explanation": "what unnecessary changes are in the bloated version"
}}

Return ONLY valid JSON.
"""


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class ReviewPair:
    """A synthesized review comment with minimal diff for SFT."""
    repo: str
    pr_number: Optional[int]
    language: str
    title: str
    diff: str
    persona: str
    review_type: str
    file_path: str
    comment: str
    impact: str
    minimal_fix: str
    test_for_fix: str
    quality_score: float
    backend: str


@dataclass
class DPOPair:
    """A DPO preference pair: minimal diff vs. bloated diff."""
    repo: str
    pr_number: Optional[int]
    language: str
    review_comment: str
    chosen: str
    rejected: str
    chosen_explanation: str
    rejected_explanation: str


# ── Quality Scoring ───────────────────────────────────────────────────────────

def score_review(result: dict, diff: str) -> float:
    """Score review quality (0.0 – 1.0)."""
    score = 0.3

    comment = result.get("comment", "")
    minimal_fix = result.get("minimal_fix", "")
    test_for_fix = result.get("test_for_fix", "")
    impact = result.get("impact", "")

    # Has substantive comment
    if len(comment) > 80:
        score += 0.15

    # Has file path or code reference
    if result.get("file_path") or re.search(r'`\w', comment):
        score += 0.1

    # Has minimal fix
    if minimal_fix and len(minimal_fix) > 30:
        score += 0.2

    # Has test
    if test_for_fix and len(test_for_fix) > 30:
        score += 0.1

    # Has impact explanation
    if impact and len(impact) > 30:
        score += 0.1

    # Review references something in the diff
    diff_identifiers = set(re.findall(r'\b\w{3,}\b', diff[:2000]))
    comment_identifiers = set(re.findall(r'\b\w{3,}\b', comment))
    overlap = len(diff_identifiers & comment_identifiers)
    if overlap >= 5:
        score += 0.05

    return round(min(1.0, score), 3)


# ── Synthesizer Class ─────────────────────────────────────────────────────────

class ReviewSynthesizer:
    """
    Persona-based code review synthesizer.

    For each PR diff, generates review comments from multiple reviewer perspectives.
    Supports both Claude and vLLM backends.
    """

    def __init__(
        self,
        input_path: Path,
        output_path: Path,
        backend: str = "claude",
        vllm_urls: Optional[list[str]] = None,
        vllm_model: str = "Qwen/Qwen2.5-72B-Instruct",
        personas: Optional[list[str]] = None,
        workers: int = 20,
        min_quality_score: float = 0.5,
        generate_dpo: bool = False,
    ) -> None:
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.vllm_urls = vllm_urls or []
        self.vllm_model = vllm_model
        self.personas = personas or list(REVIEWER_PERSONAS.keys())
        self.workers = workers
        self.min_quality_score = min_quality_score
        self.generate_dpo = generate_dpo
        self._semaphore = asyncio.Semaphore(workers)
        # Use itertools.cycle for safe, monotonic round-robin URL selection
        self._vllm_cycle = itertools.cycle(vllm_urls or ["placeholder"])
        self._stats = {"total": 0, "success": 0, "failed": 0, "filtered": 0}
        self._write_lock = asyncio.Lock()

        if backend == "claude":
            _api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not _api_key:
                raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set.")
            self._claude = anthropic.AsyncAnthropic(api_key=_api_key)

        if generate_dpo:
            dpo_stem = self.output_path.stem + "_dpo"
            self.dpo_path = self.output_path.parent / f"{dpo_stem}.jsonl"

    async def _call_claude(self, system: str, user: str, max_tokens: int = 1500) -> Optional[str]:
        """Call Claude API."""
        try:
            resp = await self._claude.messages.create(
                model="claude-opus-4-6",
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                temperature=0.3,
            )
            return resp.content[0].text
        except Exception as e:
            logger.debug(f"Claude error: {e}")
            return None

    async def _call_vllm(self, system: str, user: str, max_tokens: int = 1500) -> Optional[str]:
        """Call vLLM server."""
        if not self.vllm_urls:
            return None

        url = next(self._vllm_cycle)

        payload = {
            "model": self.vllm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.3,
            "max_tokens": max_tokens,
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
                        logger.debug(f"vLLM {resp.status}")
                        return None
        except Exception as e:
            logger.debug(f"vLLM error: {e}")
            return None

    async def _call_llm(self, system: str, user: str, max_tokens: int = 1500) -> Optional[str]:
        """Route to configured backend."""
        if self.backend == "claude":
            return await self._call_claude(system, user, max_tokens)
        return await self._call_vllm(system, user, max_tokens)

    def _extract_json(self, response: str) -> Optional[dict]:
        """Extract JSON from LLM response."""
        if not response:
            return None
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        code_match = re.search(r'```(?:json)?\s*\n(.*?)```', response, re.DOTALL)
        if code_match:
            try:
                return json.loads(code_match.group(1))
            except json.JSONDecodeError:
                pass
        # Try to find any JSON object in the response
        obj_match = re.search(r'\{.*\}', response, re.DOTALL)
        if obj_match:
            try:
                return json.loads(obj_match.group(0))
            except json.JSONDecodeError:
                pass
        return None

    async def _synthesize_review(self, pr: dict, persona: str) -> Optional[ReviewPair]:
        """Synthesize a review comment for a single PR using one persona."""
        diff = pr.get("diff", "")
        if not diff or len(diff) < 50:
            return None

        persona_system = REVIEWER_PERSONAS[persona]
        persona_context = f"Review as a {persona} expert. Focus on {persona}-related issues."

        user_prompt = REVIEW_SYNTHESIS_USER_TEMPLATE.format(
            repo=pr.get("repo", "unknown/repo"),
            language=pr.get("language", "python"),
            title=pr.get("title", ""),
            diff=diff[:5000],
            persona_context=persona_context,
        )

        response = await self._call_llm(persona_system, user_prompt)
        if not response:
            return None

        result = self._extract_json(response)
        if not result:
            return None

        quality = score_review(result, diff)
        if quality < self.min_quality_score:
            return None

        return ReviewPair(
            repo=pr.get("repo", ""),
            pr_number=pr.get("pr_number"),
            language=pr.get("language", "python"),
            title=pr.get("title", ""),
            diff=diff[:5000],
            persona=persona,
            review_type=result.get("review_type", "advisory"),
            file_path=result.get("file_path", ""),
            comment=result.get("comment", ""),
            impact=result.get("impact", ""),
            minimal_fix=result.get("minimal_fix", ""),
            test_for_fix=result.get("test_for_fix", ""),
            quality_score=quality,
            backend=self.backend,
        )

    async def _generate_dpo_pair(self, review_pair: ReviewPair) -> Optional[DPOPair]:
        """Generate a DPO preference pair from a review pair."""
        if not review_pair.minimal_fix or len(review_pair.minimal_fix) < 20:
            return None

        user_prompt = DPO_GENERATION_USER_TEMPLATE.format(
            review_comment=review_pair.comment[:500],
            minimal_fix=review_pair.minimal_fix[:2000],
        )

        response = await self._call_llm("You are a code review training data generator.", user_prompt)
        if not response:
            return None

        result = self._extract_json(response)
        if not result or not result.get("rejected"):
            return None

        return DPOPair(
            repo=review_pair.repo,
            pr_number=review_pair.pr_number,
            language=review_pair.language,
            review_comment=review_pair.comment,
            # Always use the already-generated minimal_fix as chosen — don't trust LLM to copy it
            chosen=review_pair.minimal_fix,
            rejected=result.get("rejected", ""),
            chosen_explanation=result.get("chosen_explanation", ""),
            rejected_explanation=result.get("rejected_explanation", ""),
        )

    async def _process_pr(self, pr: dict, out_f, dpo_f=None) -> int:
        """Process one PR across all configured personas.

        The semaphore gates at the PR level only — _synthesize_review must NOT
        acquire the same semaphore (that would deadlock when all slots are held
        by outer _process_pr calls waiting for inner slots to free).
        """
        async with self._semaphore:
            saved = 0
            for persona in self.personas:
                review = await self._synthesize_review(pr, persona)
                if review:
                    self._stats["success"] += 1
                    async with self._write_lock:
                        await out_f.write(json.dumps(asdict(review)) + "\n")
                    saved += 1

                    if dpo_f and review.minimal_fix:
                        dpo_pair = await self._generate_dpo_pair(review)
                        if dpo_pair:
                            async with self._write_lock:
                                await dpo_f.write(json.dumps(asdict(dpo_pair)) + "\n")
                else:
                    self._stats["filtered"] += 1

            self._stats["total"] += 1
            return saved

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

    async def synthesize_all(self) -> int:
        """Synthesize reviews for all PRs."""
        logger.info(f"Loading PRs from {self.input_path}")
        prs = self._load_prs()
        logger.info(
            f"Synthesizing {len(prs):,} PRs × {len(self.personas)} personas "
            f"= {len(prs) * len(self.personas):,} reviews using {self.backend} backend"
        )

        async with aiofiles.open(self.output_path, "w") as out_f:
            if self.generate_dpo:
                async with aiofiles.open(self.dpo_path, "w") as dpo_f:
                    tasks = [self._process_pr(pr, out_f, dpo_f) for pr in prs]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
            else:
                tasks = [self._process_pr(pr, out_f) for pr in prs]
                results = await asyncio.gather(*tasks, return_exceptions=True)

        total = sum(r for r in results if isinstance(r, int))

        logger.success(
            f"Synthesis complete: {total:,} reviews written to {self.output_path}\n"
            f"  Success: {self._stats['success']:,}\n"
            f"  Filtered (low quality): {self._stats['filtered']:,}"
        )
        return total


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Synthesize persona-based code reviews")
    parser.add_argument("--input", required=True, help="Input JSONL with raw PR diffs")
    parser.add_argument("--output", default="data/synthesized/review_pairs.jsonl")
    parser.add_argument("--backend", choices=["claude", "vllm"], default="claude")
    parser.add_argument(
        "--vllm-urls", nargs="+", default=None,
        help="vLLM server URLs (e.g. http://localhost:8001/v1)",
    )
    parser.add_argument(
        "--personas", nargs="+", default=None,
        choices=list(REVIEWER_PERSONAS.keys()),
        help="Reviewer personas to use (default: all)",
    )
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--min-score", type=float, default=0.5)
    parser.add_argument("--generate-dpo", action="store_true")
    args = parser.parse_args()

    synthesizer = ReviewSynthesizer(
        input_path=Path(args.input),
        output_path=Path(args.output),
        backend=args.backend,
        vllm_urls=args.vllm_urls,
        personas=args.personas,
        workers=args.workers,
        min_quality_score=args.min_score,
        generate_dpo=args.generate_dpo,
    )
    n = asyncio.run(synthesizer.synthesize_all())
    print(f"\nTotal reviews synthesized: {n:,}")
