# MergePilot — Roadmap

## Current Status: v1 (In Development)

---

## v1 — REVIEW (Target: Q2 2026)

Core review capability: structured review comments with implementation PR.

### Milestones

- [x] Architecture design and data source identification
- [x] Training pipeline design (SFT → GRPO → DPO)
- [ ] GitHub PR crawler for top 50k repos
- [ ] Engineering blog crawler
- [ ] Review taxonomy (25 categories)
- [ ] Review synthesizer
- [ ] PR generator
- [ ] Stage 1 SFT on 200k review pairs
- [ ] Stage 2 GRPO with merge outcome reward
- [ ] Stage 3 DPO on review quality
- [ ] MergeBench evaluation harness
- [ ] GitHub App deployment

### v1 Target Metrics

| Metric | Target |
|--------|--------|
| Review precision (real bugs found) | >70% |
| Review recall (no major bugs missed) | >80% |
| PR acceptance rate | >40% |
| Merge prediction AUC | >0.75 |
| Code quality delta | >+0.3 |

---

## v1.5 — SPECIALIZE (Target: Q3 2026)

Language-specific specialization: Python, TypeScript, Go, Rust, Java.

### Milestones

- [ ] Per-language training data splits
- [ ] Language-specific review taxonomy extensions
- [ ] Framework-specific agents (Django, React, FastAPI, Next.js)
- [ ] Test coverage enforcement agent
- [ ] Security-focused review mode (OWASP Top 10)
- [ ] Benchmarks per language category

---

## v2 — PIPELINE (Target: Q4 2026)

Full CI/CD integration: review → improve → test → merge pipeline.

### Milestones

- [ ] CI/CD integration (GitHub Actions, CircleCI)
- [ ] Automated test generation for reviewed code
- [ ] Regression detection before PR opens
- [ ] Review → implement → test → merge loop with zero human intervention
- [ ] Team adaptation: learn each team's specific style and standards

---

## v3 — ENTERPRISE (Target: Q1 2027)

Enterprise deployment with team-level knowledge and compliance checking.

### Milestones

- [ ] On-premise deployment option
- [ ] SOC 2 compliance checking (no secrets, no PII in code)
- [ ] Internal codebase learning (adapt to company's conventions)
- [ ] Review analytics dashboard
- [ ] Integration with Jira, Linear, GitHub Projects

---

## Long-Term Vision

- **Zero-latency review**: under 3 seconds from PR open to first review comment
- **Language coverage**: every major programming language + major frameworks
- **Trust**: earn maintainer trust to auto-merge PRs below a risk threshold
- **Formal verification**: integrate with Lean 4 for provably correct code transformations
