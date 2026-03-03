# MergePilot Architecture

## Overview

MergePilot is a 7B specialist model trained end-to-end on review-to-merge outcomes. The core insight: existing code review tools are trained on code generation. MergePilot is trained on what happens _after_ a review comment — did the code change? Did the PR merge? Did tests pass?

This document covers the full technical architecture: data pipeline, training approach, reward signal design, and 90-day execution plan.

---

## Data Pipeline

### Source: GitHub PR Outcome Pairs

We crawl the top 50k GitHub repositories (by stars, activity, and commit frequency) and extract structured review-to-merge triples:

```
(review_comment, resulting_diff, merge_outcome)
```

Where `merge_outcome` = {merged_with_approval, merged_after_changes, closed_without_merge, still_open}.

We keep only `merged_with_approval` and `merged_after_changes` pairs for positive training signal. We sample `closed_without_merge` pairs as negative examples (review generated churn but no quality improvement).

**Scale**: 400k high-quality pairs from 50k repos across Python, TypeScript, Go, Rust, Java.

### Quality Filters

- PR must have at least 2 reviewer approvals
- Minimum 1 test file changed or added in resulting diff
- Diff must be under 400 lines (scope discipline signal)
- Repository must have CI configured (so we can verify test pass)
- Review comment must have been directly addressed (not just a suggestion)

### Data Schema

```json
{
  "repo": "owner/repo",
  "pr_number": 123,
  "review_comment": "This function has a race condition when multiple goroutines ...",
  "review_author": "maintainer_username",
  "review_type": "blocking",
  "resulting_diff": "--- a/pkg/cache/cache.go\n+++ b/pkg/cache/cache.go\n...",
  "diff_lines_added": 12,
  "diff_lines_removed": 8,
  "tests_added": true,
  "ci_passed": true,
  "pr_merged": true,
  "days_to_merge": 2,
  "reviewer_approved_immediately": true
}
```

---

## Training Approach

### Base Model

**Qwen2.5-7B-Coder-Instruct** — chosen for:
- Strong code generation baseline
- Context window supports large diffs (32k tokens)
- Instruct-tuned for chat-style interaction
- Efficient enough for RL rollouts on 18× A6000

### Hardware: 18× NVIDIA A6000

- 48GB VRAM per card = 864GB total
- DeepSpeed ZeRO-3 for parameter sharding
- LoRA rank 64 for parameter-efficient training
- Gradient accumulation for effective batch size = 256

### Stage 1: Supervised Fine-Tuning (SFT)

**Duration**: ~6 hours on 18× A6000

**Data**: 400k (review_comment, repo_context, resulting_diff) triples

**Format**:
```
<|im_start|>system
You are MergePilot, an expert code reviewer that turns review comments into merged PRs.
Given a review comment and repository context, generate the minimal diff that addresses
the review, plus tests that prove the fix. Output: <diff>...</diff><tests>...</tests>
<|im_end|>
<|im_start|>user
Repository: {repo}
Review comment: {comment}
File context: {file_content}
<|im_end|>
<|im_start|>assistant
<think>{reasoning}</think>
<diff>{minimal_diff}</diff>
<tests>{test_code}</tests>
<|im_end|>
```

**LoRA Config**:
- Rank: 64
- Alpha: 128
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Dropout: 0.05

### Stage 2: RL with Verifiable Reward (GRPO)

**Duration**: ~4 hours on 18× A6000

**The core technical novelty**: PR merge outcome is a free, verifiable reward signal — the same insight that made DeepSeek-R1 work applied to code review.

**Reward function** (composite, each component 0.0–1.0):

```python
reward = (
    0.40 * merge_reward        # PR accepted and merged by maintainer
  + 0.35 * test_reward         # All tests pass after applying diff
  + 0.15 * regression_reward   # No existing tests broken
  + 0.10 * scope_reward        # Diff size ≤ 1.3× the minimum viable fix
)
```

- `merge_reward`: Apply diff to a sandbox branch, open simulated PR, check if it passes a trained merge-likelihood classifier (trained on 400k merge outcomes)
- `test_reward`: Execute `pytest` / `go test` / `cargo test` in sandbox environment after applying diff
- `regression_reward`: Run original test suite on modified code, check all pass
- `scope_reward`: Compare diff size to gold-standard diff size from training data

**GRPO Config**:
- Num generations per prompt: 8 (sample 8 diffs, reward best)
- Learning rate: 5e-6
- Epochs: 1 over RL training set
- Gradient accumulation: 8

### Stage 3: DPO Alignment

**Duration**: ~2 hours

**Purpose**: Enforce scope discipline — teach MergePilot to prefer minimal diffs over comprehensive rewrites.

**Preference pairs** (constructed from Stage 1 training data):

| Chosen | Rejected |
|--------|----------|
| 8-line fix addressing exactly the review comment | 80-line refactor that also "improves" surrounding code |
| Fix with 2 targeted tests | Fix with 15 tests covering unrelated edge cases |
| Error message fix without touching error handling pattern | Error message fix that also restructures error handling |

**Beta**: 0.1 (conservative — we don't want to over-constrain the model)

---

## Reward Signal Design

The key challenge in code review RL is defining what "good" means. We use a four-component reward:

### Component 1: Merge Reward (40%)

A classifier trained on 400k PR outcomes predicts merge likelihood given (review comment, diff, repo context). Score is continuous 0–1. This is the primary signal — did the change satisfy the reviewer?

### Component 2: Test Reward (35%)

Execute the generated tests in a sandboxed environment. Binary: 1.0 if all pass, 0.0 if any fail. This enforces that MergePilot writes real tests, not placeholder assertions.

### Component 3: Regression Reward (15%)

Run the repository's existing test suite after applying the generated diff. Penalizes changes that break existing behavior. Continuous: fraction of existing tests still passing.

### Component 4: Scope Reward (10%)

```python
scope_ratio = len(generated_diff) / len(gold_diff)
scope_reward = max(0, 1 - max(0, scope_ratio - 1.0))
```

Full reward if diff is ≤ gold size. Decreasing reward as diff grows beyond gold size. Encourages minimal targeted changes.

---

## 90-Day Execution Plan

### Days 1–20: Data Collection

- Set up GitHub API crawling infrastructure (50 worker threads)
- Collect 400k PR outcome pairs from top 50k repos
- Filter, deduplicate, and quality-score all pairs
- Build sandbox execution environment for test reward computation
- Collect repo context: README, CONTRIBUTING, existing test patterns

### Days 21–45: Synthesis and Data Preparation

- Synthesize (review comment, repo context, diff, reasoning) triples using Qwen2.5-72B
- Extract reviewer preference patterns per maintainer (for preference modeling)
- Build DPO pair dataset: (minimal diff, bloated diff) from real PR examples
- Construct RL task set: PRs where we can execute sandbox tests
- Quality filter: remove pairs where gold diff doesn't pass tests

### Days 46–70: Training

- Stage 1 SFT: 6 hours on 18× A6000
- Stage 2 GRPO: 4 hours (1 epoch, 8 rollouts per prompt)
- Stage 3 DPO: 2 hours
- Checkpoint evaluation at each stage
- WandB tracking throughout

### Days 71–90: Evaluation and Deployment

- MergeBench evaluation on 50 held-out repositories
- A/B comparison: MergePilot vs. GPT-4 scaffolded vs. human reviewer
- Measure: merge rate, test pass rate, first-approval rate, diff size ratio
- Deploy review agent as GitHub App
- Public release of model weights and training code

---

## Evaluation: MergeBench

50 held-out repositories not seen during training, stratified by:
- Language (Python, TypeScript, Go, Rust, Java)
- Repository size (small <100 PRs, medium 100–1000 PRs, large >1000 PRs)
- Codebase domain (web, systems, ML, data, devtools)

For each repository, we sample 20 historical review comments and measure:

| Metric | Description |
|--------|-------------|
| Merge rate | % of generated PRs that would be accepted |
| Test pass rate | % of generated PRs where all tests pass |
| First-approval rate | % of PRs where simulated reviewer approves immediately |
| Regression rate | % of PRs that break existing tests |
| Scope ratio | Avg generated diff size / gold diff size |
| Time to merge | Simulated days from review comment to merge |

---

## Differentiators

1. **Trained on outcomes, not syntax** — reward is merge outcome, not code correctness metrics
2. **Scope discipline as first-class objective** — DPO alignment explicitly penalizes over-engineering
3. **Test generation is required** — reward is zero if generated PR has no tests
4. **Per-project preference modeling** — learns maintainer patterns from repo history
5. **Free verifiable reward** — no human annotation needed; GitHub merge data is the label
