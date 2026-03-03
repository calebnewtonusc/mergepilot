# MergePilot — Data Sources

## Overview

MergePilot trains on 4 data streams totaling 200k+ review interaction pairs. The key insight: we track the full outcome loop — not just what the reviewer said, but whether the code changed in response and whether the PR merged.

---

## Stream 1: GitHub PR Review Comments from Top 50k Repos (50%)

### Source
- GitHub's public event stream (GH Archive)
- REST API: `/repos/{owner}/{repo}/pulls/{pull_number}/reviews`
- `/repos/{owner}/{repo}/pulls/{pull_number}/comments`

### Repo Selection Criteria
- Stars: ≥500
- Language: Python, TypeScript, JavaScript, Go, Rust, Java, C++
- Activity: last commit within 12 months
- Review density: ≥5 reviews per PR on average

### Data Points per PR
- PR diff (unified diff format)
- Review comments (inline + overall)
- Code changes made in response to reviews
- Final PR outcome: merged / closed without merge
- CI/CD result: tests pass / fail
- Time to merge (hours)

### Format
```json
{
  "conversations": [
    {"role": "system", "content": "You are an expert code reviewer..."},
    {"role": "user", "content": "PR diff:\n```diff\n{diff}\n```\nContext: {repo_context}"},
    {"role": "assistant", "content": "{structured_review}"}
  ],
  "metadata": {
    "repo": "pallets/flask",
    "pr_number": 4823,
    "merged": true,
    "review_type": "security",
    "taxonomy_labels": ["sql_injection", "input_validation"],
    "merge_time_hours": 48
  }
}
```

### Quality Filters
- Reviewer must have ≥100 contributions to the repo
- Review must have ≥2 sentences
- Code change must follow within 72 hours (actionable review)
- PR must be either merged or explicitly closed (not abandoned)

### Scale
- ~5M raw review comments from GitHub Archive
- ~400k filtered (quality filters applied)
- ~100k synthesized into training format

---

## Stream 2: Engineering Blog Posts on Code Review (20%)

### Sources
- **Google Engineering Practices**: google.github.io/eng-practices
- **Airbnb Engineering Blog**: medium.com/airbnb-engineering
- **Netflix Tech Blog**: netflixtechblog.com
- **Stripe Engineering**: stripe.com/blog/engineering
- **Shopify Engineering**: shopify.engineering
- **Meta Engineering**: engineering.fb.com
- **LinkedIn Engineering**: engineering.linkedin.com
- **DoorDash Engineering**: doordash.engineering
- **Uber Engineering**: eng.uber.com
- **Dropbox Tech Blog**: dropbox.tech

### Content Types
- Code review process guides
- "What makes a good PR" articles
- Common code review anti-patterns
- Language/framework-specific guidelines

### Format Conversion
Blog posts → Socratic review dialogues: "reviewer asks, author explains, reviewer suggests fix"

### Scale
- ~500 engineering blog posts
- ~40k synthesized review dialogues

---

## Stream 3: Open Source Style Guides and Review Guidelines (15%)

### Sources
- **PEP 8** (Python) + Google Python Style Guide
- **Google TypeScript Style Guide**
- **Effective Go** (Google)
- **Rust API Guidelines**
- **Java Google Style Guide**
- **Airbnb JavaScript Style Guide**
- Major framework contribution guides (React, Django, FastAPI, Spring, etc.)

### Format
Style guide violations → review comment pairs:
```json
{
  "violation": "Function has 150 lines",
  "rule": "Functions should do one thing. If it's over ~40 lines, consider splitting.",
  "review_comment": "This function is doing three separate things: parsing, validation, and transformation. Consider splitting into parse_input(), validate_schema(), and transform_data() to improve readability and testability.",
  "fix_approach": "Extract to three separate functions"
}
```

### Scale
- ~50 style guides and guidelines documents
- ~30k review comment examples

---

## Stream 4: Synthesized Review → Implementation Pairs (15%)

### Generation Pipeline
For each (code diff, review comment) pair from Stream 1:
1. Have Qwen2.5-72B generate an "improvement PR" that implements the review
2. Validate syntactically (Python: `ast.parse()`, JS/TS: `tsc --noEmit`)
3. Check the diff is minimal (implements exactly what was requested)
4. Filter pairs where the generated fix is plausible

### Review Taxonomy Coverage
Ensure each of the 25 review taxonomy categories is represented:
- Correctness bugs (null pointer, off-by-one, etc.)
- Security issues (injection, XSS, auth bypass)
- Performance bottlenecks
- API design issues
- Test coverage gaps
- Documentation missing
- Naming conventions
- Error handling
- Type safety
- Concurrency issues
... (25 total)

### Scale
- ~30k synthesized (review → implementation PR) pairs
- Validated for syntactic correctness

---

## Data Pipeline Summary

```
GitHub API + GH Archive → discovery/ crawlers → data/raw/
Engineering blogs       → discovery/         → data/raw/
Style guides            → discovery/         → data/raw/
All raw data            → synthesis/          → data/synthesized/
                        → deduplication      → data/train/
                        → quality filter     → data/final/
```

### Final Dataset Stats

| Split | PRs | Review Pairs | Validated |
|-------|-----|--------------|-----------|
| Train | 180k | 180,000 | 90% |
| Val | 10k | 10,000 | 100% |
| Test (MergeBench) | 500 | held-out | 100% |
| DPO pairs | — | 20,000 | 100% |

---

## Legal and Ethical Notes

- GitHub data: public repos under open source licenses, used per GitHub Terms
- Engineering blogs: publicly available content, used for research
- No private code: only public repositories included
- No credentials/secrets in training data: filtered out by pattern matching
- GDPR: no personal data in training examples (code content only)
