"""
PR Author Agent — Implements code review suggestions as concrete code changes

Given a (diff, review_comment) pair, generates the minimal patch that
addresses the review suggestion.

Used for:
  - Training data synthesis (30k implementation pairs)
  - Live "auto-fix" feature in the MergePilot API
"""

import os
import re
from dataclasses import dataclass
from typing import Optional

import anthropic
from loguru import logger

from core.review_taxonomy import classify_review_comment
from synthesis.prompts import PR_GENERATION_SYSTEM, PR_GENERATION_USER


@dataclass
class ImplementationResult:
    """Result of implementing a review comment."""
    pr_id: str
    original_diff: str
    review_comment: str
    implementation_diff: str
    commit_message: str
    category: str
    syntax_valid: bool
    confidence: float  # 0-1


class PRAuthorAgent:
    """
    Implements review suggestions as code changes.

    In production: uses the fine-tuned MergePilot model.
    During development: falls back to Claude claude-opus-4-6.
    """

    def __init__(
        self,
        vllm_url: Optional[str] = None,
        backend: str = "claude",
    ) -> None:
        self.vllm_url = vllm_url
        self.backend = backend

        if backend == "claude":
            self._claude = anthropic.Anthropic(
                api_key=os.environ["ANTHROPIC_API_KEY"]
            )
            logger.info("PRAuthorAgent initialized with Claude backend")

    def implement(
        self,
        original_diff: str,
        review_comment: str,
        pr_id: str = "unknown",
    ) -> Optional[ImplementationResult]:
        """
        Implement a review comment as a code change.

        Args:
            original_diff: The unified diff that was reviewed
            review_comment: The specific review comment to implement
            pr_id: PR identifier for tracking

        Returns:
            ImplementationResult with the patch diff, or None if failed
        """
        if not original_diff or not review_comment:
            return None

        prompt = PR_GENERATION_USER.format(
            original_diff=original_diff[:3000],
            review_comment=review_comment[:500],
        )

        response = self._call_model(PR_GENERATION_SYSTEM, prompt, max_tokens=800)
        if not response:
            return None

        implementation_diff = self._extract_diff(response)
        commit_message = self._extract_commit_message(response)
        syntax_valid = self._check_diff_syntax(implementation_diff)
        confidence = self._estimate_confidence(implementation_diff, review_comment)

        return ImplementationResult(
            pr_id=pr_id,
            original_diff=original_diff[:2000],
            review_comment=review_comment,
            implementation_diff=implementation_diff,
            commit_message=commit_message,
            category=classify_review_comment(review_comment),
            syntax_valid=syntax_valid,
            confidence=confidence,
        )

    def implement_batch(
        self,
        pairs: list[dict],
    ) -> list[Optional[ImplementationResult]]:
        """Implement multiple (diff, review) pairs synchronously."""
        results = []
        for pair in pairs:
            result = self.implement(
                original_diff=pair.get("diff", ""),
                review_comment=pair.get("review_comment", ""),
                pr_id=pair.get("pr_id", "unknown"),
            )
            results.append(result)
        return results

    def format_as_training_pair(self, result: ImplementationResult) -> Optional[dict]:
        """Format an implementation result as a ShareGPT training pair."""
        if not result.syntax_valid or result.confidence < 0.5:
            return None

        return {
            "conversations": [
                {
                    "role": "system",
                    "content": "You are MergePilot. Implement the code review suggestion.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Original change:\n```diff\n{result.original_diff}\n```\n\n"
                        f"Review: {result.review_comment}"
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        f"```diff\n{result.implementation_diff}\n```\n\n"
                        f"Commit: {result.commit_message}"
                    ),
                },
            ],
            "metadata": {
                "pr_id": result.pr_id,
                "category": result.category,
                "type": "pr_implementation",
                "syntax_valid": result.syntax_valid,
                "confidence": result.confidence,
            },
        }

    # -------------------------------------------------------------------------
    # Private methods
    # -------------------------------------------------------------------------

    def _call_model(self, system: str, user: str, max_tokens: int) -> Optional[str]:
        if self.backend == "claude":
            return self._call_claude(system, user, max_tokens)
        return self._call_vllm(system, user, max_tokens)

    def _call_claude(self, system: str, user: str, max_tokens: int) -> Optional[str]:
        try:
            resp = self._claude.messages.create(
                model="claude-opus-4-6",
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return resp.content[0].text
        except Exception as e:
            logger.error(f"Claude error: {e}")
            return None

    def _call_vllm(self, system: str, user: str, max_tokens: int) -> Optional[str]:
        import httpx
        try:
            resp = httpx.post(
                f"{self.vllm_url}/v1/chat/completions",
                json={
                    "model": "/model",
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.3,
                },
                headers={"Authorization": f"Bearer {os.getenv('VLLM_API_KEY', '')}"},
                timeout=60.0,
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"vLLM error: {e}")
        return None

    def _extract_diff(self, response: str) -> str:
        """Extract unified diff from model response."""
        match = re.search(r"```diff\n(.*?)\n```", response, re.DOTALL)
        if match:
            return match.group(1)
        # Try without language specifier
        match = re.search(r"```\n([-+@].*?)\n```", response, re.DOTALL)
        if match:
            return match.group(1)
        return ""

    def _extract_commit_message(self, response: str) -> str:
        """Extract commit message from response."""
        match = re.search(r"(?:commit|fix|feat|refactor):\s*(.+)", response, re.IGNORECASE)
        if match:
            return match.group(0).strip()[:100]
        return "fix: implement review suggestion"

    def _check_diff_syntax(self, diff: str) -> bool:
        """Validate that the diff looks like a real unified diff."""
        if not diff or len(diff) < 10:
            return False

        lines = diff.strip().split("\n")
        has_additions = any(
            line.startswith("+") and not line.startswith("+++")
            for line in lines
        )
        has_enough_lines = len(lines) >= 3

        return has_additions and has_enough_lines

    def _estimate_confidence(self, diff: str, review_comment: str) -> float:
        """
        Estimate confidence that the implementation addresses the review.

        Simple heuristic: check if keywords from the review appear in the diff.
        """
        if not diff:
            return 0.0

        confidence = 0.5

        # Longer, more structured diffs are more confident
        lines = diff.split("\n")
        if len(lines) > 5:
            confidence += 0.1
        if len(lines) > 10:
            confidence += 0.1

        # Check if diff contains words from the review
        review_words = set(
            w.lower() for w in re.findall(r"\b\w{4,}\b", review_comment)
        )
        diff_words = set(
            w.lower() for w in re.findall(r"\b\w{4,}\b", diff)
        )
        overlap = len(review_words & diff_words) / max(len(review_words), 1)
        confidence += min(0.3, overlap * 0.3)

        return min(1.0, confidence)
