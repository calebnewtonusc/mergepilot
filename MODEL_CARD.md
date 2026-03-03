# MergePilot — Model Card

## Model Details

| Field | Value |
|-------|-------|
| **Model name** | MergePilot v1 |
| **Base model** | Qwen/Qwen2.5-7B-Coder-Instruct |
| **Fine-tuning method** | LoRA (SFT) → GRPO RL → DPO |
| **Parameters** | 7.6B (base) + LoRA adapters |
| **Context window** | 8,192 tokens |
| **Languages** | Python, TypeScript, JavaScript, Go, Rust, Java, C++ |
| **License** | Apache 2.0 |
| **Developer** | Caleb Newton (calebnewtonusc) |
| **Version** | 1.0.0 |
| **Release date** | Q2 2026 (projected) |

---

## Training Details

### Stage 1: Supervised Fine-Tuning
- **Algorithm**: SFT via TRL SFTTrainer
- **Data**: 180,000 review interaction pairs (4 streams)
- **LoRA rank**: 64, alpha: 128, dropout: 0.05
- **Target modules**: q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj
- **Learning rate**: 2e-4 with WarmupDecayLR
- **Epochs**: 3
- **Hardware**: 18× A6000 48GB, DeepSpeed ZeRO-3

### Stage 2: RL with Merge Outcome Reward
- **Algorithm**: GRPO (Group Relative Policy Optimization)
- **Reward signal**: PR merge outcome — +1.0 if PR merged, -1.0 if closed without merge
- **Partial rewards**: +0.3 for tests passing without merge, +0.5 for code quality improvement
- **KL penalty**: 0.02 vs. SFT checkpoint
- **Group size N**: 4 completions per prompt

### Stage 3: DPO on Review Quality
- **Algorithm**: DPO (Direct Preference Optimization), beta=0.1
- **Chosen**: Actionable review that leads to code improvement
- **Rejected**: Vague/nitpicky review that doesn't improve code quality
- **Preference signals**: Review acted upon (code changed), PR merged after review

---

## Intended Use

### Primary Use Cases
- **Code review automation**: Reviewing PRs in GitHub repositories
- **Improvement PR generation**: Implementing review suggestions as ready-to-merge PRs
- **Merge prediction**: Estimating whether a PR will be accepted
- **Security audit**: Identifying security issues in code changes
- **Code quality improvement**: Suggesting refactors, performance improvements

### Out-of-Scope Uses
- **Not a static analyzer**: Complements but does not replace ESLint, SonarQube, etc.
- **Not for secrets detection**: Use dedicated tools (truffleHog, git-secrets)
- **Not legal advice**: Comments about licensing compliance are informational only
- **Not for obfuscated/malicious code**: Designed for legitimate software development

---

## Evaluation (v1 Targets)

| Benchmark | Metric | Target | Status |
|-----------|--------|--------|--------|
| MergeBench-Review | Review precision | >70% | In training |
| MergeBench-Review | Review recall | >80% | In training |
| MergeBench-PR | PR acceptance rate | >40% | In training |
| MergeBench-Predict | Merge prediction AUC | >0.75 | In training |
| MergeBench-Quality | Code quality delta | >+0.3 | In training |

---

## Limitations

1. **Context window**: 8,192 tokens limits review of very large diffs (>500 lines)
2. **Language coverage**: Optimized for top 7 languages; less effective on niche languages
3. **Private codebase conventions**: Cannot learn team-specific conventions without fine-tuning
4. **Test coverage**: Cannot run code to verify correctness — reviews are static
5. **Knowledge cutoff**: Trained on code through 2025; newer frameworks/APIs may not be known

---

## Ethical Considerations

### Code Quality
MergePilot may introduce bugs in generated improvement PRs. All generated PRs must be reviewed by a human before merging.

### False Negatives
The model may miss security vulnerabilities. It is not a substitute for dedicated security auditing tools.

### Bias in Training Data
The training corpus skews toward popular, well-maintained repositories. Code from niche domains or languages may receive lower-quality reviews.

---

## Citation

```bibtex
@inproceedings{newton2026mergepilot,
  title     = {MergePilot: Learning to Review and Improve Code from Merge Outcomes},
  author    = {Newton, Caleb and others},
  booktitle = {ICSE 2026 Workshop on AI for Software Engineering},
  year      = {2026},
}
```
