# MergePilot

**Code review that ships.**

MergePilot is the world's first specialist model for code review automation — trained specifically to turn review comments into merged, test-backed pull requests. Unlike general-purpose LLMs that generate review comments nobody acts on, MergePilot is trained on the full review-to-merge outcome: what comment was left, what code changed in response, whether the PR merged, and whether tests passed.

**Target**: Review comments become merged PRs. Every time.

---

## Why MergePilot Is Different

Every existing code review tool takes the same approach: run static analysis + prompt GPT-4 to describe what it found. CodeRabbit generates comments. Copilot suggests inline changes. Neither is trained on what actually causes code to change.

MergePilot's approach: train a specialist on outcomes. The reward signal is `PR merged + tests pass + no regression + minimal diff` — the same free verifiable reward that made DeepSeek-R1 work, applied to code review.

```
GPT-4 + static analysis scaffolding  →  comments that get dismissed
Trained MergePilot specialist          →  review turns into merged PR
```

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                        MERGEPILOT SYSTEM                             │
│                                                                      │
│  Review Comment ──► Intent Classifier ──► Resolution Strategy        │
│         │                   │                       │                │
│         │          [Blocking / Advisory /      [Fix / Refactor /     │
│         │           Style / Security]           Test / Docs]         │
│         ▼                   ▼                       ▼                │
│  Repo Context ────► Code Generator ────► Diff Minimizer              │
│         │                   │                       │                │
│         └───────────────────┴───────────────────────┘                │
│                             │                                        │
│                      Test Generator                                  │
│                             │                                        │
│                     [PR Submission]                                  │
│                             │                                        │
│              Merge Outcome Verifier ◄── RL Reward Signal             │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/calebnewtonusc/mergepilot
cd mergepilot
pip install -r requirements.txt
cp .env.example .env  # Fill in API keys

# Run full pipeline (data → training → eval), ~48 hours on 18× A6000
python pipeline.py

# Or step by step:
python pipeline.py --stage discovery    # Collect PR outcome pairs from GitHub
python pipeline.py --stage synthesis    # Synthesize training data
python pipeline.py --stage train        # SFT + RL + DPO (3-stage training)
python pipeline.py --stage eval         # MergeBench evaluation
```

---

## Run on a Repository

```bash
# Point MergePilot at any GitHub PR
python agents/review_agent.py \
  --repo "owner/repo" \
  --pr 123

# Or use the API
curl -X POST http://localhost:8000/review \
  -H "Content-Type: application/json" \
  -d '{"repo": "owner/repo", "pr_number": 123}'
```

---

## Performance Targets (v1)

| Metric | Target | GPT-4 scaffolded (baseline) |
|--------|--------|-----------------------------|
| PR merge rate (after MergePilot action) | >70% | ~25% |
| Test pass rate on generated PRs | >90% | ~40% |
| Reviewer approval on first submission | >65% | ~20% |
| Avg diff size vs. manual fix | <1.3× | ~2.8× |
| No regression rate | >95% | ~75% |

---

## Repo Structure

```
mergepilot/
├── web/                        # Coming-soon landing page (Next.js 15)
│   └── src/app/page.tsx
├── discovery/
│   └── github_pr_outcomes.py   # Crawl GitHub PRs for review-to-merge pairs
├── synthesis/
│   └── review_synthesizer.py   # Synthesize training pairs from PR data
├── training/
│   ├── train.py                # Stage 1: SFT on review-to-PR pairs
│   ├── train_rl.py             # Stage 2: GRPO with merge-outcome reward
│   └── configs/
│       └── ds_config.json      # DeepSpeed ZeRO-3 config
├── evaluation/
│   └── mergebench.py           # Evaluation: merge rate, test pass, diff size
├── agents/
│   └── review_agent.py         # Agentic loop: review comment → opened PR
├── pipeline.py                 # Master orchestration script
├── requirements.txt
├── .env.example
└── ARCHITECTURE.md
```

---

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — Data pipeline, training approach, reward design, 90-day plan

---

## Hardware Requirements

- **Training**: 18× NVIDIA A6000 (48GB each) = 864GB total VRAM
- **Synthesis**: Azure burst for PR data synthesis (Qwen2.5-72B)
- **Inference**: 1× A6000 per active review session

---

## License

MIT License — open training pipeline, open weights (post v1 release).
