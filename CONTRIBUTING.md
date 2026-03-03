# Contributing to MergePilot

## Ways to Contribute

1. **Data quality**: Add PR datasets from underrepresented languages or domains
2. **Review taxonomy**: Expand the 25-category taxonomy with new review types
3. **Evaluation**: Add test cases to MergeBench
4. **Agents**: Improve reviewer, PR author, or merge predictor agents
5. **Language support**: Add language-specific parsers and quality metrics

## Development Setup

```bash
git clone https://github.com/calebnewtonusc/mergepilot.git
cd mergepilot
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add GITHUB_TOKEN, ANTHROPIC_API_KEY
```

## Code Standards

- Python 3.11+, type hints on all functions
- `loguru` for logging (no `print()`)
- `dataclasses` for structured data
- Tests in `tests/` using pytest

## Commit Messages

```
feat: add github_pr_crawler with rate limit handling
fix: handle deleted files in pr diff parsing
data: add 10k Python security review examples
eval: add SQL injection detection to mergebench
```

## PR Template

```
## What this PR does
[1-2 sentence summary]

## Data impact (if applicable)
- PRs added: N
- Review comments added: N

## Testing
- [ ] pytest tests pass
- [ ] check_env.sh passes
```

## License

Contributions licensed under MIT (code) and CC BY 4.0 (data).
