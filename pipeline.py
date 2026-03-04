"""
MergePilot Master Pipeline
Orchestrates the full data → training → evaluation pipeline.
~48 hours total on 18× A6000 + Azure burst for synthesis.

Usage:
  python pipeline.py                          # Full pipeline
  python pipeline.py --stage discovery        # Step 1: collect PR outcome pairs
  python pipeline.py --stage synthesis        # Step 2: generate training pairs
  python pipeline.py --stage train            # Step 3: 3-stage training
  python pipeline.py --stage eval             # Step 4: MergeBench
"""

import subprocess

import typer
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer()


STAGES = [
    # ── Discovery ────────────────────────────────────────────────────────
    {
        "name": "check_env",
        "description": "Verify environment, GPU setup, and GitHub API token",
        "cmd": "python -c \"import torch; print(f'GPUs: {torch.cuda.device_count()}')\"",
        "phase": "discovery",
        "estimated_hours": 0.05,
    },
    {
        "name": "discover_prs",
        "description": "Crawl GitHub top 50k repos for PR outcome pairs",
        "cmd": "python discovery/github_pr_outcomes.py --output-dir data/raw/pr_pairs --repos 50000",
        "phase": "discovery",
        "estimated_hours": 8.0,
    },
    {
        "name": "filter_pairs",
        "description": "Filter pairs: merged, test-covered, minimal diff, CI verified",
        "cmd": "python discovery/github_pr_outcomes.py --filter-only --input data/raw/pr_pairs --output data/filtered",
        "phase": "discovery",
        "estimated_hours": 1.0,
    },
    # ── Synthesis ────────────────────────────────────────────────────────
    {
        "name": "start_vllm",
        "description": "Launch Qwen2.5-72B synthesis servers (Azure burst)",
        "cmd": "bash scripts/start_vllm.sh",
        "phase": "synthesis",
        "estimated_hours": 0.5,
    },
    {
        "name": "synthesize_pairs",
        "description": "Synthesize (review, reasoning, diff, tests) training triples",
        "cmd": "python synthesis/review_synthesizer.py --input data/filtered/pr_pairs.jsonl --output data/synthesized/review_pairs.jsonl --workers 32",
        "phase": "synthesis",
        "estimated_hours": 16.0,
    },
    {
        "name": "build_dpo_pairs",
        "description": "Build (minimal-scope, bloated-scope) DPO preference pairs",
        "cmd": "python synthesis/review_synthesizer.py --input data/filtered/pr_pairs.jsonl --output data/synthesized/dpo_pairs.jsonl --generate-dpo",
        "phase": "synthesis",
        "estimated_hours": 4.0,
    },
    {
        "name": "build_rl_tasks",
        "description": "Build RL task set: PRs with executable sandbox tests",
        "cmd": "python synthesis/pr_generator.py --backend claude --workers 20",
        "phase": "synthesis",
        "estimated_hours": 2.0,
    },
    # ── Training ─────────────────────────────────────────────────────────
    {
        "name": "train_sft",
        "description": "Stage 1: SFT on 400k review-to-PR pairs (6h on 18× A6000)",
        "cmd": "torchrun --nproc_per_node=18 training/train.py",
        "phase": "train",
        "estimated_hours": 6.0,
    },
    {
        "name": "train_rl",
        "description": "Stage 2: GRPO with merge-outcome reward (4h on 18× A6000)",
        "cmd": "torchrun --nproc_per_node=18 training/train_rl.py",
        "phase": "train",
        "estimated_hours": 4.0,
    },
    {
        "name": "train_dpo",
        "description": "Stage 3: DPO on (minimal, bloated) scope preference pairs (2h)",
        "cmd": "torchrun --nproc_per_node=18 training/train_dpo.py",
        "phase": "train",
        "estimated_hours": 2.0,
    },
    # ── Evaluation ───────────────────────────────────────────────────────
    {
        "name": "mergebench",
        "description": "MergeBench: merge rate, test pass rate, diff size on 50 held-out repos",
        "cmd": "python evaluation/mergebench.py",
        "phase": "eval",
        "estimated_hours": 4.0,
    },
]


def run_stage(stage: dict, dry_run: bool = False) -> bool:
    """Execute a pipeline stage. Returns True on success."""
    console.print(f"\n[bold cyan]> {stage['name']}[/bold cyan]: {stage['description']}")
    console.print(f"  [dim]{stage['cmd']}[/dim]")

    if dry_run:
        console.print("  [yellow](dry run — skipping)[/yellow]")
        return True

    result = subprocess.run(stage["cmd"], shell=True)
    if result.returncode != 0:
        console.print(f"  [red]Failed (exit {result.returncode})[/red]")
        return False

    console.print("  [green]Complete[/green]")
    return True


@app.command()
def main(
    stage: str = typer.Option(
        None, help="Run only this phase: discovery | synthesis | train | eval"
    ),
    from_stage: str = typer.Option(None, help="Resume pipeline from this stage name"),
    dry_run: bool = typer.Option(False, help="Print commands without executing"),
    list_stages: bool = typer.Option(False, "--list", help="List all stages and exit"),
):
    """MergePilot: full training pipeline from raw GitHub data to deployed model."""

    if list_stages:
        table = Table(title="MergePilot Pipeline Stages")
        table.add_column("Stage", style="cyan")
        table.add_column("Phase")
        table.add_column("Description")
        table.add_column("Est. Hours", justify="right")
        for s in STAGES:
            table.add_row(
                s["name"], s["phase"], s["description"], str(s["estimated_hours"])
            )
        console.print(table)

        total = sum(s["estimated_hours"] for s in STAGES)
        console.print(f"\nTotal estimated: {total:.1f} hours")
        return

    # Filter stages
    stages_to_run = STAGES
    if stage:
        stages_to_run = [s for s in STAGES if s["phase"] == stage]
        if not stages_to_run:
            console.print(f"[red]Unknown phase: {stage}[/red]")
            raise typer.Exit(1)
    elif from_stage:
        names = [s["name"] for s in STAGES]
        if from_stage not in names:
            console.print(f"[red]Unknown stage: {from_stage}[/red]")
            raise typer.Exit(1)
        idx = names.index(from_stage)
        stages_to_run = STAGES[idx:]

    total_hours = sum(s["estimated_hours"] for s in stages_to_run)
    console.print(
        f"\n[bold]MergePilot Pipeline[/bold] — {len(stages_to_run)} stages, ~{total_hours:.0f}h estimated"
    )
    if dry_run:
        console.print("[yellow]DRY RUN MODE[/yellow]")

    for s in stages_to_run:
        success = run_stage(s, dry_run=dry_run)
        if not success:
            console.print(
                f"\n[red bold]Pipeline failed at stage: {s['name']}[/red bold]"
            )
            console.print(f"To resume: python pipeline.py --from-stage {s['name']}")
            raise typer.Exit(1)

    console.print("\n[green bold]Pipeline complete.[/green bold]")


if __name__ == "__main__":
    app()
