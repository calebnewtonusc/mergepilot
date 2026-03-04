"""
Reviewer Agent — Primary code review agent using the trained MergePilot model

Uses the fine-tuned model (or Claude fallback) to generate structured,
taxonomy-tagged code reviews from PR diffs.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

import anthropic
from loguru import logger

from core.review_taxonomy import TAXONOMY
from core.impact_scorer import ImpactScorer
from synthesis.prompts import MERGEPILOT_SYSTEM


@dataclass
class ReviewComment:
    """A single structured review comment."""

    category: str
    severity: str
    file_path: Optional[str]
    line_number: Optional[int]
    observation: str
    issue: str
    suggestion: str
    code_example: Optional[str]
    impact_score: float


@dataclass
class CodeReview:
    """A complete code review for a PR."""

    pr_id: str
    overall_assessment: str  # "APPROVE" | "REQUEST_CHANGES" | "COMMENT"
    summary: str
    comments: list[ReviewComment] = field(default_factory=list)
    blocking_count: int = 0
    suggestion_count: int = 0
    optional_count: int = 0
    overall_quality: float = 0.0
    model_used: str = "unknown"


class ReviewerAgent:
    """
    Primary code review agent.

    In production: loads the fine-tuned MergePilot model via vLLM.
    During development: falls back to Claude claude-opus-4-6.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        vllm_url: Optional[str] = None,
        backend: str = "claude",
        vllm_model: Optional[str] = None,
    ) -> None:
        self.model_path = model_path
        self.vllm_url = vllm_url
        self.backend = backend
        self.vllm_model = vllm_model or os.getenv(
            "MODEL_PATH", "Qwen/Qwen2.5-7B-Coder-Instruct"
        )
        self._scorer = ImpactScorer()

        if backend == "claude":
            _api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not _api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY environment variable is not set. "
                    "Export it before running: export ANTHROPIC_API_KEY=sk-ant-..."
                )
            self._claude = anthropic.Anthropic(api_key=_api_key)
            logger.info("ReviewerAgent initialized with Claude backend")
        else:
            logger.info(f"ReviewerAgent initialized with vLLM backend: {vllm_url}")

    def review(
        self,
        diff: str,
        repo: str = "unknown",
        pr_title: str = "",
        language: str = "Python",
        context: str = "",
        pr_id: str = "unknown",
    ) -> CodeReview:
        """
        Generate a structured code review for a PR diff.

        Args:
            diff: The unified diff of the PR
            repo: Repository name (owner/repo)
            pr_title: Title of the pull request
            language: Primary programming language
            context: Additional context (description, linked issues)
            pr_id: PR identifier for tracking

        Returns:
            CodeReview with structured comments and assessment
        """
        prompt = self._build_prompt(diff, repo, pr_title, language, context)
        raw_review = self._call_model(prompt)

        if not raw_review:
            return CodeReview(
                pr_id=pr_id,
                overall_assessment="COMMENT",
                summary="Unable to generate review.",
                model_used=self.backend,
            )

        return self._parse_review(raw_review, pr_id)

    def review_batch(self, prs: list[dict]) -> list[CodeReview]:
        """Review multiple PRs."""
        reviews = []
        for pr in prs:
            review = self.review(
                diff=pr.get("diff", ""),
                repo=pr.get("repo", "unknown"),
                pr_title=pr.get("title", ""),
                language=pr.get("language", "Python"),
                pr_id=f"{pr.get('repo', '')}#{pr.get('pr_number', '')}",
            )
            reviews.append(review)
        return reviews

    # -------------------------------------------------------------------------
    # Private methods
    # -------------------------------------------------------------------------

    def _build_prompt(
        self,
        diff: str,
        repo: str,
        title: str,
        language: str,
        context: str,
    ) -> str:
        """Build the review prompt."""
        return (
            f"Repository: {repo}\n"
            f"Language: {language}\n"
            f"PR Title: {title}\n\n"
            f"```diff\n{diff[:4000]}\n```\n\n"
            f"Additional context: {context}\n\n"
            "Provide a thorough code review following the structured format."
        )

    def _call_model(self, prompt: str) -> Optional[str]:
        """Call the review model."""
        if self.backend == "claude":
            return self._call_claude(prompt)
        return self._call_vllm(prompt)

    def _call_claude(self, prompt: str) -> Optional[str]:
        try:
            resp = self._claude.messages.create(
                model="claude-opus-4-6",
                max_tokens=1500,
                system=MERGEPILOT_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            block = resp.content[0]
            return block.text if hasattr(block, "text") else None
        except Exception as e:
            logger.error(f"Claude error: {e}")
            return None

    def _call_vllm(self, prompt: str) -> Optional[str]:
        import httpx

        try:
            resp = httpx.post(
                f"{self.vllm_url}/v1/chat/completions",
                json={
                    "model": self.vllm_model,
                    "messages": [
                        {"role": "system", "content": MERGEPILOT_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 1500,
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

    def _parse_review(self, raw_review: str, pr_id: str) -> CodeReview:
        """Parse raw review text into structured CodeReview."""
        import re

        # Determine overall assessment
        assessment = "COMMENT"
        if re.search(r"\bAPPROVE\b|\bLGTM\b|\bapprove\b", raw_review, re.IGNORECASE):
            assessment = "APPROVE"
        elif re.search(
            r"REQUEST_CHANGES\b|blocking|critical", raw_review, re.IGNORECASE
        ):
            assessment = "REQUEST_CHANGES"

        # Extract summary (first paragraph)
        paragraphs = [p.strip() for p in raw_review.split("\n\n") if p.strip()]
        summary = paragraphs[0][:300] if paragraphs else "Review completed."

        # Parse individual comments
        comments = self._extract_comments(raw_review)

        # Count by severity
        blocking = sum(1 for c in comments if c.severity == "blocking")
        suggestion = sum(1 for c in comments if c.severity == "suggestion")
        optional = sum(1 for c in comments if c.severity == "optional")

        overall_quality = self._scorer.score_review(raw_review)

        return CodeReview(
            pr_id=pr_id,
            overall_assessment=assessment,
            summary=summary,
            comments=comments,
            blocking_count=blocking,
            suggestion_count=suggestion,
            optional_count=optional,
            overall_quality=overall_quality,
            model_used=self.backend,
        )

    def _extract_comments(self, raw_review: str) -> list[ReviewComment]:
        """Extract structured comments from raw review text."""
        import re

        comments = []

        # Look for structured comment blocks
        category_pattern = re.compile(
            r"\[([A-Z_]+)\]\s*(?:File:\s*(\S+))?,?\s*(?:Line:\s*(\d+))?",
            re.IGNORECASE,
        )

        # Split into comment sections
        sections = re.split(r"\n(?=\[)", raw_review)

        for section in sections:
            if not section.strip():
                continue

            # Try to extract structured fields
            cat_match = category_pattern.search(section)
            category_tag = cat_match.group(1).lower() if cat_match else None
            file_path = cat_match.group(2) if cat_match else None
            line_num_str = cat_match.group(3) if cat_match else None
            line_number = int(line_num_str) if line_num_str else None

            # Map category tag to taxonomy ID
            category = self._map_category_tag(category_tag)
            cat_obj = TAXONOMY.get(category)
            severity = cat_obj.severity if cat_obj else "optional"

            # Extract fields
            obs_match = re.search(
                r"Observation:\s*(.+?)(?:\n|Issue:)", section, re.DOTALL
            )
            issue_match = re.search(
                r"Issue:\s*(.+?)(?:\n|Suggestion:)", section, re.DOTALL
            )
            sug_match = re.search(r"Suggestion:\s*(.+?)(?:\n```|$)", section, re.DOTALL)
            code_match = re.search(r"```\w*\n(.*?)\n```", section, re.DOTALL)

            observation = obs_match.group(1).strip() if obs_match else section[:100]
            issue = issue_match.group(1).strip() if issue_match else ""
            suggestion = sug_match.group(1).strip() if sug_match else ""
            code_example = code_match.group(1) if code_match else None

            if len(observation) < 10:
                continue

            impact = self._scorer.score_comment(section).total

            comments.append(
                ReviewComment(
                    category=category,
                    severity=severity,
                    file_path=file_path,
                    line_number=line_number,
                    observation=observation,
                    issue=issue,
                    suggestion=suggestion,
                    code_example=code_example,
                    impact_score=impact,
                )
            )

        return sorted(comments, key=lambda c: c.impact_score, reverse=True)

    def _map_category_tag(self, tag: Optional[str]) -> str:
        """Map a review tag like 'security' to taxonomy ID."""
        if not tag:
            return "general"

        mapping = {
            "correctness": "correctness_bug",
            "security": "security_injection",
            "performance": "performance_n_plus_1",
            "api_design": "api_naming",
            "tests": "test_missing",
            "docs": "docs_missing",
            "error_handling": "error_handling",
            "type_safety": "type_safety",
            "style": "style_formatting",
            "architecture": "architecture",
        }

        tag_lower = tag.lower()
        if tag_lower in mapping:
            return mapping[tag_lower]
        # If the tag itself is a known taxonomy ID use it directly;
        # otherwise fall back to "general" (calling classify_review_comment
        # with a single word produces unreliable keyword matches).
        return tag_lower if tag_lower in TAXONOMY else "general"
