"""
Merge Predictor Agent — Predicts whether a PR will be merged

Uses:
  - Code diff quality metrics
  - Review comment sentiment and severity
  - PR metadata (size, title, description)

Output: merge probability 0-1, recommendation, risk factors
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import anthropic
from loguru import logger

from core.impact_scorer import ImpactScorer
from synthesis.prompts import MERGE_PREDICTION_SYSTEM, MERGE_PREDICTION_USER


@dataclass
class MergePrediction:
    """Prediction result for a PR."""
    pr_id: str
    merge_probability: float   # 0.0 - 1.0
    recommendation: str        # "merge" | "request_changes" | "close"
    key_factors: list[str] = field(default_factory=list)
    risk_factors: list[str] = field(default_factory=list)
    confidence: float = 0.5
    model_used: str = "unknown"


class MergePredictorAgent:
    """
    Predicts PR merge probability using model + heuristics.

    Used for:
    - RL reward signal: merged PRs get +1.0, closed PRs get -1.0
    - Live API endpoint: /v1/predict
    - Training data quality filter
    """

    # Thresholds for heuristic fallback
    HIGH_MERGE_THRESHOLD = 0.7
    LOW_MERGE_THRESHOLD = 0.3

    def __init__(
        self,
        vllm_url: Optional[str] = None,
        backend: str = "heuristic",
    ) -> None:
        self.vllm_url = vllm_url
        self.backend = backend
        self._scorer = ImpactScorer()

        if backend == "claude":
            _api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not _api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY environment variable is not set. "
                    "Export it before running: export ANTHROPIC_API_KEY=sk-ant-..."
                )
            self._claude = anthropic.Anthropic(api_key=_api_key)
            logger.info("MergePredictorAgent initialized with Claude backend")
        else:
            logger.info("MergePredictorAgent initialized with heuristic backend")

    def predict(
        self,
        diff: str,
        review_comments: list[str],
        pr_id: str = "unknown",
        merged: Optional[bool] = None,
    ) -> MergePrediction:
        """
        Predict merge probability for a PR.

        Args:
            diff: Unified diff of the PR
            review_comments: List of review comment strings
            pr_id: PR identifier
            merged: Ground truth (for evaluation only)

        Returns:
            MergePrediction with probability and recommendation
        """
        # Always compute heuristic features
        features = self._scorer.compute_merge_probability_features(
            diff=diff,
            review_text="\n\n".join(review_comments),
        )

        if self.backend == "claude" and diff and review_comments:
            return self._predict_with_model(diff, review_comments, features, pr_id)

        return self._predict_heuristic(features, pr_id)

    def predict_batch(self, prs: list[dict]) -> list[MergePrediction]:
        """Predict merge probability for a batch of PRs."""
        predictions = []
        for pr in prs:
            pred = self.predict(
                diff=pr.get("diff", ""),
                review_comments=[
                    c.get("body", "") if isinstance(c, dict) else str(c)
                    for c in pr.get("review_comments", [])
                ],
                pr_id=f"{pr.get('repo', '')}#{pr.get('pr_number', '')}",
                merged=pr.get("merged"),
            )
            predictions.append(pred)
        return predictions

    def compute_rl_reward(
        self,
        predicted: MergePrediction,
        actual_merged: bool,
    ) -> float:
        """
        Compute RL reward signal from merge outcome.

        +1.0 if actually merged (positive signal)
        -1.0 if closed without merge (negative signal)
        Scaled by prediction confidence.
        """
        outcome = 1.0 if actual_merged else -1.0
        # Scale by how aligned the prediction was
        if actual_merged:
            alignment = predicted.merge_probability
        else:
            alignment = 1.0 - predicted.merge_probability

        return outcome * (0.5 + 0.5 * alignment)

    # -------------------------------------------------------------------------
    # Private methods
    # -------------------------------------------------------------------------

    def _predict_with_model(
        self,
        diff: str,
        review_comments: list[str],
        features: dict,
        pr_id: str,
    ) -> MergePrediction:
        """Use LLM to predict merge probability."""
        reviews_text = "\n---\n".join(review_comments[:5])  # Max 5 comments
        prompt = MERGE_PREDICTION_USER.format(
            diff=diff[:2000],
            review_comments=reviews_text[:1000],
        )

        try:
            resp = self._claude.messages.create(
                model="claude-opus-4-6",
                max_tokens=300,
                system=MERGE_PREDICTION_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text
            # Robustly extract JSON — Claude may wrap it in markdown code fences
            result = self._extract_json(raw)

            return MergePrediction(
                pr_id=pr_id,
                merge_probability=float(result.get("merge_probability", 0.5)),
                recommendation=result.get("recommendation", "comment"),
                key_factors=result.get("key_factors", []),
                risk_factors=result.get("risk_factors", []),
                confidence=0.8,
                model_used="claude-opus-4-6",
            )
        except Exception as e:
            logger.debug(f"Model prediction failed: {e}, falling back to heuristic")
            return self._predict_heuristic(features, pr_id)

    def _extract_json(self, text: str) -> dict:
        """Robustly extract a JSON object from a text response (handles markdown fences)."""
        import re
        if not text:
            raise ValueError("Empty response")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Try JSON inside markdown code fences
        code_match = re.search(r'```(?:json)?\s*\n(.*?)```', text, re.DOTALL)
        if code_match:
            return json.loads(code_match.group(1))
        # Try any JSON object in the response
        obj_match = re.search(r'\{.*\}', text, re.DOTALL)
        if obj_match:
            return json.loads(obj_match.group(0))
        raise ValueError(f"No JSON found in response: {text[:100]}")

    def _predict_heuristic(self, features: dict, pr_id: str) -> MergePrediction:
        """Heuristic merge probability based on extracted features."""
        prob = 0.5

        # Small PRs more likely to merge
        if features.get("is_small_pr"):
            prob += 0.1
        elif features.get("is_large_pr"):
            prob -= 0.1

        # Tests increase merge probability
        if features.get("has_tests"):
            prob += 0.15

        # High review quality with no blocking issues
        review_score = features.get("review_quality_score", 0.5)
        blocking = features.get("blocking_issues", 0)

        if blocking == 0:
            prob += 0.1
        else:
            prob -= blocking * 0.15

        prob += (review_score - 0.5) * 0.2

        # Complexity increase is a risk
        complexity = features.get("complexity_delta", 0)
        if complexity > 1.0:
            prob -= 0.05

        prob = max(0.05, min(0.95, prob))

        # Recommendation based on probability
        if prob >= self.HIGH_MERGE_THRESHOLD:
            recommendation = "merge"
        elif prob <= self.LOW_MERGE_THRESHOLD:
            recommendation = "request_changes"
        else:
            recommendation = "comment"

        # Build factor lists
        key_factors = []
        risk_factors = []

        if features.get("is_small_pr"):
            key_factors.append("Small focused PR")
        if features.get("has_tests"):
            key_factors.append("Includes test coverage")
        if features.get("has_docs"):
            key_factors.append("Documentation updated")

        if blocking > 0:
            risk_factors.append(f"{blocking} blocking review comment(s)")
        if features.get("is_large_pr"):
            risk_factors.append("Large PR — harder to review")
        if complexity > 1.0:
            risk_factors.append(f"Increased complexity (+{complexity:.1f})")

        return MergePrediction(
            pr_id=pr_id,
            merge_probability=round(prob, 3),
            recommendation=recommendation,
            key_factors=key_factors,
            risk_factors=risk_factors,
            confidence=0.6,
            model_used="heuristic",
        )
