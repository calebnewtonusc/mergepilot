"""
Review Pair Synthesizer
Uses Qwen2.5-72B via vLLM to enrich raw PR outcome pairs with:
  - Structured reasoning (what the review was asking for and why)
  - Minimal diff annotation (what is the gold-standard minimal change)
  - DPO preference pairs (minimal vs. bloated diff)
  - RL task construction (executable sandbox tasks)

Concurrency: 32 parallel synthesis workers against vLLM server.
"""

import asyncio
import json
import os
from pathlib import Path

import aiohttp
from loguru import logger
from tqdm.asyncio import tqdm_asyncio


SYNTHESIS_PROMPT = """\
You are analyzing a GitHub pull request review comment and the resulting merged diff.

Repository: {repo}
Language: {language}
Review comment: {review_comment}

Code context (diff hunk where comment was left):
{file_context}

Full PR diff (what was merged):
{diff}

Extract the following as a JSON object:
{{
  "review_type": "blocking|advisory|style|security|performance|test",
  "review_intent": "1-2 sentence description of what the reviewer was asking for",
  "minimal_diff": "The minimal unified diff that would address this specific review comment (not the full PR diff)",
  "test_code": "Test code that verifies the fix from minimal_diff",
  "reasoning": "Why this specific change addresses the review comment — the key insight",
  "scope_discipline": "What about this change shows good scope discipline (not doing too much or too little)"
}}

Return ONLY valid JSON. minimal_diff should be smaller than the full PR diff.
"""

DPO_SYNTHESIS_PROMPT = """\
You are generating a DPO (Direct Preference Optimization) training pair for a code review model.

Review comment: {review_comment}
Language: {language}
Minimal correct fix: {minimal_diff}

Generate a "bloated" version of this fix that addresses the comment but includes unnecessary additional changes:
- Extra refactoring not requested
- Additional helper functions that aren't needed
- Rewriting surrounding code that works fine
- Adding comments/docs that weren't requested

Return JSON:
{{
  "chosen_diff": "{minimal_diff}",
  "rejected_diff": "...(the bloated version)...",
  "chosen_reason": "why the minimal diff is preferred",
  "rejected_reason": "why the bloated diff is worse despite technically working"
}}

Return ONLY valid JSON.
"""


async def synthesize_pair(
    session: aiohttp.ClientSession,
    vllm_url: str,
    api_key: str,
    pair: dict,
) -> dict | None:
    """Enrich a single PR outcome pair with synthesized reasoning and minimal diff."""
    prompt = SYNTHESIS_PROMPT.format(
        repo=pair.get("repo", "owner/repo"),
        language=pair.get("language", "python"),
        review_comment=pair.get("review_comment", ""),
        file_context=pair.get("file_context", "")[:2000],
        diff=pair.get("diff", "")[:4000],
    )

    payload = {
        "model": "Qwen/Qwen2.5-72B-Instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 2048,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        async with session.post(
            f"{vllm_url}/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            text = data["choices"][0]["message"]["content"]
            enriched = json.loads(text)
            # Merge original pair fields with synthesized enrichment
            return {**pair, **enriched}
    except Exception as e:
        logger.debug(f"Synthesis failed for {pair.get('repo')}#{pair.get('pr_number')}: {e}")
        return None


async def synthesize_dpo_pair(
    session: aiohttp.ClientSession,
    vllm_url: str,
    api_key: str,
    pair: dict,
) -> dict | None:
    """Generate a DPO preference pair (minimal vs. bloated diff)."""
    if not pair.get("minimal_diff"):
        return None

    prompt = DPO_SYNTHESIS_PROMPT.format(
        review_comment=pair.get("review_comment", ""),
        language=pair.get("language", "python"),
        minimal_diff=pair.get("minimal_diff", ""),
    )

    payload = {
        "model": "Qwen/Qwen2.5-72B-Instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,  # Higher temp for creative bloated version
        "max_tokens": 2048,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        async with session.post(
            f"{vllm_url}/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            text = data["choices"][0]["message"]["content"]
            dpo = json.loads(text)
            return {
                "repo": pair.get("repo"),
                "pr_number": pair.get("pr_number"),
                "language": pair.get("language"),
                "review_comment": pair.get("review_comment"),
                "chosen_diff": dpo.get("chosen_diff", pair.get("minimal_diff", "")),
                "rejected_diff": dpo.get("rejected_diff", ""),
                "chosen_reason": dpo.get("chosen_reason", ""),
                "rejected_reason": dpo.get("rejected_reason", ""),
            }
    except Exception as e:
        logger.debug(f"DPO synthesis failed: {e}")
        return None


async def synthesize_all(
    input_path: Path,
    output_path: Path,
    vllm_url: str,
    api_key: str,
    mode: str = "pairs",
    concurrency: int = 32,
) -> None:
    """Synthesize all pairs in parallel with bounded concurrency."""
    with open(input_path) as f:
        pairs = [json.loads(line) for line in f if line.strip()]

    logger.info(f"Synthesizing {len(pairs)} pairs → {output_path} (mode: {mode})")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_synthesize(session: aiohttp.ClientSession, pair: dict) -> dict | None:
        async with semaphore:
            if mode == "dpo":
                return await synthesize_dpo_pair(session, vllm_url, api_key, pair)
            elif mode == "rl_tasks":
                # For RL tasks, we need pairs with repo_path for sandbox execution
                # Just pass through pairs that have enough context
                if pair.get("diff") and pair.get("review_comment"):
                    return {
                        "review_comment": pair.get("review_comment"),
                        "file_context": pair.get("file_context", ""),
                        "language": pair.get("language", "python"),
                        "repo": pair.get("repo"),
                        "pr_number": pair.get("pr_number"),
                        "gold_diff": pair.get("minimal_diff", pair.get("diff", "")),
                        "repo_path": None,  # Set during local execution
                    }
                return None
            else:
                return await synthesize_pair(session, vllm_url, api_key, pair)

    async with aiohttp.ClientSession() as session:
        tasks = [bounded_synthesize(session, p) for p in pairs]
        results = await tqdm_asyncio.gather(*tasks, desc=f"Synthesizing ({mode})", return_exceptions=True)

    valid = [r for r in results if isinstance(r, dict)]
    logger.info(f"Synthesis complete: {len(valid)}/{len(pairs)} succeeded")

    with open(output_path, "w") as f:
        for pair in valid:
            f.write(json.dumps(pair) + "\n")


if __name__ == "__main__":
    import typer

    def main(
        input: str = "./data/filtered/filtered_pairs.jsonl",
        output: str = "./data/synthesized/review_pairs.jsonl",
        vllm_url: str = None,
        concurrency: int = 32,
        mode: str = "pairs",
    ):
        url = vllm_url or os.environ.get("VLLM_SYNTHESIS_URL")
        key = os.environ.get("VLLM_API_KEY")
        if not url:
            raise ValueError("VLLM_SYNTHESIS_URL not set.")
        if not key:
            raise ValueError("VLLM_API_KEY not set.")

        asyncio.run(synthesize_all(
            Path(input), Path(output), url, key, mode=mode, concurrency=concurrency
        ))

    typer.run(main)
