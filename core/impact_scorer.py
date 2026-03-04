"""
Impact Scorer — Quantifies the quality and impact of code review comments

Scores comments on:
  - Specificity (does it cite a file/line?)
  - Actionability (does it suggest a fix?)
  - Severity (blocking vs. suggestion vs. optional)
  - Educational value (does it explain the principle?)
"""

import re
from dataclasses import dataclass
from typing import Optional

from core.review_taxonomy import (
    TAXONOMY,
    classify_review_comment,
    get_blocking_categories,
)


@dataclass
class ImpactScore:
    """Scored review comment."""

    comment: str
    category: str
    specificity: float  # 0-1: does it cite location?
    actionability: float  # 0-1: does it suggest a fix?
    severity_weight: float  # 1.0 blocking, 0.6 suggestion, 0.3 optional
    educational: float  # 0-1: explains the why?
    total: float


@dataclass
class DiffQualityMetrics:
    """Metrics for a code diff."""

    lines_added: int
    lines_removed: int
    files_changed: int
    has_tests: bool
    has_docs: bool
    complexity_delta: float  # estimated cyclomatic complexity change
    coverage_delta: float  # estimated coverage change (-1.0 to +1.0)


class ImpactScorer:
    """
    Scores review comments and code diffs for quality and impact.

    Used to:
    1. Filter low-quality training data
    2. Compute DPO rewards
    3. Weight training examples by importance
    """

    SEVERITY_WEIGHTS = {
        "blocking": 1.0,
        "suggestion": 0.6,
        "optional": 0.3,
    }

    def score_comment(self, comment: str) -> ImpactScore:
        """Score a single review comment."""
        category = classify_review_comment(comment)
        cat_obj = TAXONOMY.get(category)
        severity_weight = self.SEVERITY_WEIGHTS.get(
            cat_obj.severity if cat_obj else "optional", 0.3
        )

        specificity = self._score_specificity(comment)
        actionability = self._score_actionability(comment)
        educational = self._score_educational(comment)

        # Weighted average
        total = (
            0.25 * specificity
            + 0.35 * actionability
            + 0.25 * severity_weight
            + 0.15 * educational
        )

        return ImpactScore(
            comment=comment,
            category=category,
            specificity=specificity,
            actionability=actionability,
            severity_weight=severity_weight,
            educational=educational,
            total=round(total, 3),
        )

    def score_review(self, review_text: str) -> float:
        """Score an entire review (multi-comment)."""
        comments = self._split_comments(review_text)
        if not comments:
            return 0.0

        scores = [self.score_comment(c) for c in comments]
        avg_total = sum(s.total for s in scores) / len(scores)

        # Bonus for having multiple high-quality comments
        breadth_bonus = min(0.1, len(scores) * 0.02)

        # Bonus for catching blocking issues
        blocking_cats = set(get_blocking_categories())
        has_blocking = any(s.category in blocking_cats for s in scores)
        blocking_bonus = 0.1 if has_blocking else 0.0

        return min(1.0, avg_total + breadth_bonus + blocking_bonus)

    def score_diff(self, diff: str) -> DiffQualityMetrics:
        """Extract quality metrics from a code diff."""
        lines = diff.split("\n")
        added = [l for l in lines if l.startswith("+") and not l.startswith("+++")]
        removed = [l for l in lines if l.startswith("-") and not l.startswith("---")]
        file_headers = [l for l in lines if l.startswith("+++")]

        has_tests = any(
            re.search(r"test_|_test\.|spec\.|\.test\.", l, re.IGNORECASE)
            for l in file_headers
        )
        has_docs = any(
            re.search(r"README|\.md|docstring|\"\"\"", l, re.IGNORECASE) for l in added
        )

        complexity_delta = self._estimate_complexity_delta(added, removed)
        coverage_delta = 0.1 if has_tests else -0.05

        return DiffQualityMetrics(
            lines_added=len(added),
            lines_removed=len(removed),
            files_changed=len(file_headers),
            has_tests=has_tests,
            has_docs=has_docs,
            complexity_delta=complexity_delta,
            coverage_delta=coverage_delta,
        )

    def compute_merge_probability_features(
        self,
        diff: str,
        review_text: str,
        pr_size: Optional[str] = None,
    ) -> dict:
        """
        Compute features for merge probability prediction.

        Returns feature dict usable by merge predictor model.
        """
        diff_metrics = self.score_diff(diff)
        review_score = self.score_review(review_text)
        review_comments = self._split_comments(review_text)
        comment_scores = [self.score_comment(c) for c in review_comments]

        blocking_cats = set(get_blocking_categories())
        blocking_issues = sum(1 for s in comment_scores if s.category in blocking_cats)

        return {
            "lines_added": diff_metrics.lines_added,
            "lines_removed": diff_metrics.lines_removed,
            "files_changed": diff_metrics.files_changed,
            "has_tests": int(diff_metrics.has_tests),
            "has_docs": int(diff_metrics.has_docs),
            "complexity_delta": diff_metrics.complexity_delta,
            "review_quality_score": review_score,
            "num_review_comments": len(review_comments),
            "blocking_issues": blocking_issues,
            "is_small_pr": int(diff_metrics.lines_added < 100),
            "is_large_pr": int(diff_metrics.lines_added > 500),
        }

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _score_specificity(self, comment: str) -> float:
        """Does the comment cite a specific location?"""
        score = 0.0
        if re.search(
            r"(?:file|path|line|function|method|class)[\s:]+\S", comment, re.IGNORECASE
        ):
            score += 0.5
        if re.search(r"```", comment):
            score += 0.3
        if re.search(r"L\d+|line \d+", comment, re.IGNORECASE):
            score += 0.2
        return min(1.0, score)

    def _score_actionability(self, comment: str) -> float:
        """Does the comment suggest a concrete fix?"""
        score = 0.0
        action_patterns = [
            r"(?:should|could|try|use|replace|change|add|remove|fix|refactor)",
            r"(?:instead of|rather than|prefer|consider)",
            r"```",  # code example
            r"example:",
        ]
        for pattern in action_patterns:
            if re.search(pattern, comment, re.IGNORECASE):
                score += 0.25

        return min(1.0, score)

    def _score_educational(self, comment: str) -> float:
        """Does the comment explain the underlying principle?"""
        score = 0.0
        edu_patterns = [
            r"because|reason|why|principle|pattern|best practice",
            r"this (?:causes|leads to|results in|means)",
            r"(?:performance|security|maintainability) (?:issue|problem|concern)",
        ]
        for pattern in edu_patterns:
            if re.search(pattern, comment, re.IGNORECASE):
                score += 0.33

        return min(1.0, score)

    def _split_comments(self, review_text: str) -> list[str]:
        """Split a multi-comment review into individual comments."""
        # Split on category tags or double newlines
        parts = re.split(
            r"\n{2,}|\[(?:CORRECTNESS|SECURITY|PERFORMANCE|API_DESIGN|TESTS|DOCS|ERROR_HANDLING|TYPE_SAFETY|STYLE|ARCHITECTURE)\]",
            review_text,
        )
        return [p.strip() for p in parts if p.strip() and len(p.strip()) > 30]

    def _estimate_complexity_delta(self, added: list[str], removed: list[str]) -> float:
        """
        Rough cyclomatic complexity delta.
        Each branch keyword in added = +0.1 complexity.
        Each branch keyword removed = -0.1 complexity.
        """
        branch_pattern = re.compile(
            r"\b(?:if|elif|else|for|while|try|except|case|switch|&&|\|\|)\b"
        )
        added_branches = sum(len(branch_pattern.findall(l)) for l in added)
        removed_branches = sum(len(branch_pattern.findall(l)) for l in removed)
        return round((added_branches - removed_branches) * 0.1, 2)
