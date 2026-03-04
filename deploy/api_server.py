"""
MergePilot API Server

FastAPI server exposing the trained MergePilot model as a REST API.

Endpoints:
  POST /v1/review    — Full PR review (diff + metadata)
  POST /v1/fix       — Implement a review comment as a patch
  POST /v1/predict   — Predict merge probability
  GET  /health       — Health check with model info
  GET  /v1/taxonomy  — Return the review taxonomy

Usage:
  uvicorn deploy.api_server:app --host 0.0.0.0 --port 8000 --workers 4
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from loguru import logger
from pydantic import BaseModel, Field

from core.review_taxonomy import TAXONOMY
from core.impact_scorer import ImpactScorer


# ─── Auth ───────────────────────────────────────────────────────────────────

security = HTTPBearer(auto_error=False)
API_KEYS = set(os.environ.get("MERGEPILOT_API_KEYS", "").split(",")) - {""}


def verify_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    if not API_KEYS:
        return  # No auth required if no keys configured
    if not credentials or credentials.credentials not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )


# ─── State ──────────────────────────────────────────────────────────────────

_agents: dict = {}
_scorer = ImpactScorer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize agents on startup."""
    logger.info("Starting MergePilot API server...")

    backend = os.environ.get("MERGEPILOT_BACKEND", "claude")
    vllm_url = os.environ.get("VLLM_URL", "http://localhost:8001")

    # Lazy imports to keep startup fast
    from agents.reviewer_agent import ReviewerAgent
    from agents.pr_author_agent import PRAuthorAgent
    from agents.merge_predictor_agent import MergePredictorAgent

    _agents["reviewer"] = ReviewerAgent(vllm_url=vllm_url, backend=backend)
    _agents["author"] = PRAuthorAgent(vllm_url=vllm_url, backend=backend)
    _agents["predictor"] = MergePredictorAgent(vllm_url=vllm_url, backend="heuristic")

    logger.info(f"Agents loaded. Backend: {backend}")
    yield

    logger.info("Shutting down MergePilot API server.")


# ─── App ────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MergePilot API",
    description="AI-powered code review: Review -> Improve -> Merge. Automated.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # Cannot use allow_credentials=True with allow_origins=["*"] per CORS spec
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response Models ───────────────────────────────────────────────

class ReviewRequest(BaseModel):
    diff: str = Field(..., min_length=10, description="Unified diff of the PR")
    repo: str = Field("unknown", description="Repository name (owner/repo)")
    pr_title: str = Field("", description="PR title")
    language: str = Field("Python", description="Primary programming language")
    context: str = Field("", description="Additional context (description, linked issues)")
    pr_id: str = Field("", description="PR identifier for tracking")


class FixRequest(BaseModel):
    diff: str = Field(..., min_length=10, description="Original PR diff")
    review_comment: str = Field(..., min_length=10, description="Review comment to implement")
    pr_id: str = Field("", description="PR identifier")


class PredictRequest(BaseModel):
    diff: str = Field(..., min_length=10, description="PR diff")
    review_comments: list[str] = Field(default_factory=list, description="Review comments")
    pr_id: str = Field("", description="PR identifier")


# ─── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "healthy",
        "model": os.environ.get("MERGEPILOT_MODEL", "claude-opus-4-6"),
        "backend": os.environ.get("MERGEPILOT_BACKEND", "claude"),
        "taxonomy_categories": len(TAXONOMY),
        "timestamp": int(time.time()),
    }


@app.get("/v1/taxonomy")
async def get_taxonomy(auth=Depends(verify_api_key)):
    """Return the full review taxonomy."""
    return {
        "taxonomy": {
            cat_id: {
                "name": cat.name,
                "description": cat.description,
                "severity": cat.severity,
                "keywords": cat.keywords,
            }
            for cat_id, cat in TAXONOMY.items()
        }
    }


@app.post("/v1/review")
async def review_pr(request: ReviewRequest, auth=Depends(verify_api_key)):
    """
    Generate a structured code review for a PR diff.

    Returns taxonomy-tagged comments with severity, location, and suggested fixes.
    """
    if not _agents.get("reviewer"):
        raise HTTPException(500, "Reviewer agent not initialized")

    t0 = time.time()
    review = _agents["reviewer"].review(
        diff=request.diff,
        repo=request.repo,
        pr_title=request.pr_title,
        language=request.language,
        context=request.context,
        pr_id=request.pr_id or f"api_{int(t0)}",
    )
    elapsed = round(time.time() - t0, 2)

    return {
        "pr_id": review.pr_id,
        "assessment": review.overall_assessment,
        "summary": review.summary,
        "comments": [
            {
                "category": c.category,
                "severity": c.severity,
                "file": c.file_path,
                "line": c.line_number,
                "observation": c.observation,
                "issue": c.issue,
                "suggestion": c.suggestion,
                "code_example": c.code_example,
                "impact_score": c.impact_score,
            }
            for c in review.comments
        ],
        "stats": {
            "blocking": review.blocking_count,
            "suggestion": review.suggestion_count,
            "optional": review.optional_count,
            "quality_score": review.overall_quality,
        },
        "elapsed_seconds": elapsed,
    }


@app.post("/v1/fix")
async def implement_fix(request: FixRequest, auth=Depends(verify_api_key)):
    """
    Implement a review comment as a code patch.

    Returns a unified diff that addresses the specific review suggestion.
    """
    if not _agents.get("author"):
        raise HTTPException(500, "Author agent not initialized")

    t0 = time.time()
    result = _agents["author"].implement(
        original_diff=request.diff,
        review_comment=request.review_comment,
        pr_id=request.pr_id or f"fix_{int(t0)}",
    )
    elapsed = round(time.time() - t0, 2)

    if not result or not result.syntax_valid:
        raise HTTPException(
            422,
            detail="Could not generate a valid implementation. The review comment may be too vague.",
        )

    return {
        "pr_id": result.pr_id,
        "implementation_diff": result.implementation_diff,
        "commit_message": result.commit_message,
        "category": result.category,
        "confidence": result.confidence,
        "elapsed_seconds": elapsed,
    }


@app.post("/v1/predict")
async def predict_merge(request: PredictRequest, auth=Depends(verify_api_key)):
    """
    Predict whether a PR will be merged.

    Returns merge probability, recommendation, key factors, and risk factors.
    """
    if not _agents.get("predictor"):
        raise HTTPException(500, "Predictor agent not initialized")

    t0 = time.time()
    pred = _agents["predictor"].predict(
        diff=request.diff,
        review_comments=request.review_comments,
        pr_id=request.pr_id or f"pred_{int(t0)}",
    )
    elapsed = round(time.time() - t0, 2)

    return {
        "pr_id": pred.pr_id,
        "merge_probability": pred.merge_probability,
        "recommendation": pred.recommendation,
        "key_factors": pred.key_factors,
        "risk_factors": pred.risk_factors,
        "confidence": pred.confidence,
        "elapsed_seconds": elapsed,
    }


@app.post("/v1/score")
async def score_review(
    body: dict,
    auth=Depends(verify_api_key),
):
    """Score a review comment for quality (specificity, actionability, educational value)."""
    comment = body.get("comment", "")
    if not comment:
        raise HTTPException(422, "comment field required")

    score = _scorer.score_comment(comment)
    return {
        "category": score.category,
        "severity": score.severity_weight,
        "specificity": score.specificity,
        "actionability": score.actionability,
        "educational": score.educational,
        "total": score.total,
    }


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "deploy.api_server:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        workers=int(os.environ.get("API_WORKERS", 4)),
        log_level="info",
    )
