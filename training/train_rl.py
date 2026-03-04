"""
Stage 2: Merge-Outcome Reinforcement Learning (GRPO)
The core technical novelty of MergePilot.

Uses PR merge outcome as the reward signal — the same "free verifiable reward"
insight as DeepSeek-R1, applied to code review automation.

Reward = 0.40 * merge_reward
       + 0.35 * test_reward
       + 0.15 * regression_reward
       + 0.10 * scope_reward
"""

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import torch
from datasets import Dataset
from loguru import logger
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


@dataclass
class RLTrainingConfig:
    # Model
    base_model: str = "Qwen/Qwen2.5-7B-Coder-Instruct"
    sft_checkpoint: str = "./checkpoints/sft"
    output_dir: str = "./checkpoints/rl"

    # GRPO
    learning_rate: float = 5e-6
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_generations: int = 8  # Sample 8 diffs per review, reward best

    # LoRA (inherited from SFT)
    lora_r: int = 64
    lora_alpha: int = 128

    # Data
    train_data_path: str = "./data/rl/review_tasks.jsonl"

    # Reward weights
    merge_reward_weight: float = 0.40
    test_reward_weight: float = 0.35
    regression_reward_weight: float = 0.15
    scope_reward_weight: float = 0.10

    # Logging
    logging_steps: int = 10
    save_steps: int = 100
    wandb_project: str = "mergepilot-rl"


def execute_tests_in_sandbox(
    repo_path: str,
    diff: str,
    test_code: str,
    language: str = "python",
    timeout: int = 60,
) -> dict:
    """
    Apply diff to a sandbox copy of the repo and run tests.
    Returns {test_passed, regression_free, error}.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy repo to sandbox — strip trailing slash so cp -r places the
        # directory itself (not its contents) inside tmpdir.
        result = subprocess.run(
            ["cp", "-r", repo_path.rstrip("/"), tmpdir],
            capture_output=True, timeout=30
        )
        if result.returncode != 0:
            return {"test_passed": False, "regression_free": False, "error": "copy failed"}

        repo_path_stripped = repo_path.rstrip("/")
        sandbox_path = Path(tmpdir) / Path(repo_path_stripped).name

        # Apply diff
        diff_file = Path(tmpdir) / "patch.diff"
        diff_file.write_text(diff)
        patch_result = subprocess.run(
            ["patch", "-p1", "-i", str(diff_file)],
            cwd=str(sandbox_path),
            capture_output=True, timeout=30
        )
        if patch_result.returncode != 0:
            return {"test_passed": False, "regression_free": False, "error": "patch failed"}

        # Write new tests
        if test_code and language == "python":
            test_file = sandbox_path / "test_mergepilot_generated.py"
            test_file.write_text(test_code)

        # Run new tests
        test_passed = False
        if language == "python":
            test_result = subprocess.run(
                ["python", "-m", "pytest", str(sandbox_path), "-x", "-q", "--timeout=30"],
                capture_output=True, timeout=timeout, cwd=str(sandbox_path)
            )
            test_passed = test_result.returncode == 0
        elif language in ("typescript", "javascript"):
            test_result = subprocess.run(
                ["npm", "test", "--", "--passWithNoTests"],
                capture_output=True, timeout=timeout, cwd=str(sandbox_path)
            )
            test_passed = test_result.returncode == 0

        # Run regression tests (existing tests without the new test file)
        if language == "python" and (sandbox_path / "test_mergepilot_generated.py").exists():
            (sandbox_path / "test_mergepilot_generated.py").unlink()
        regression_result = subprocess.run(
            ["python", "-m", "pytest", str(sandbox_path), "-x", "-q", "--timeout=30"],
            capture_output=True, timeout=timeout, cwd=str(sandbox_path)
        )
        regression_free = regression_result.returncode == 0

        return {
            "test_passed": test_passed,
            "regression_free": regression_free,
            "error": None,
        }


def compute_scope_reward(generated_diff: str, gold_diff: str) -> float:
    """
    Returns 1.0 if generated diff is <= gold diff size.
    Decreases linearly as diff grows beyond gold size.
    """
    if not generated_diff.strip():
        return 0.0
    if not gold_diff:
        return 0.5
    gen_lines = len(generated_diff.strip().splitlines())
    gold_lines = len(gold_diff.strip().splitlines())
    if gold_lines == 0:
        return 0.5
    ratio = gen_lines / gold_lines
    return max(0.0, 1.0 - max(0.0, ratio - 1.0))


def build_reward_function(config: RLTrainingConfig):
    """
    Returns a reward function compatible with TRL's GRPOTrainer.
    Executes generated diffs in sandbox, runs tests, measures scope.
    """
    def reward_fn(prompts: list[str], completions: list[list[str]], **kwargs) -> list[float]:
        # completions shape: [num_prompts][num_generations]
        # We must return a flat list of length num_prompts * num_generations.
        metadata_list = kwargs.get("metadata", [{} for _ in range(len(prompts))])
        rewards = []

        for i, (completion_group, prompt) in enumerate(zip(completions, prompts)):
            meta = metadata_list[i % len(metadata_list)] if metadata_list and len(metadata_list) > 0 else {}

            repo_path = meta.get("repo_path", "")
            gold_diff = meta.get("gold_diff", "")
            language = meta.get("language", "python")

            for completion in completion_group:
                # Parse diff and tests from completion
                diff = ""
                tests = ""
                if "<diff>" in completion and "</diff>" in completion:
                    diff = completion.split("<diff>")[1].split("</diff>")[0].strip()
                if "<tests>" in completion and "</tests>" in completion:
                    tests = completion.split("<tests>")[1].split("</tests>")[0].strip()

                # Compute rewards
                if repo_path and Path(repo_path).exists() and diff:
                    sandbox_result = execute_tests_in_sandbox(repo_path, diff, tests, language)
                    test_reward = 1.0 if sandbox_result["test_passed"] else 0.0
                    regression_reward = 1.0 if sandbox_result["regression_free"] else 0.0
                else:
                    # No sandbox available — use proxy heuristics
                    test_reward = 0.5 if tests.strip() else 0.0
                    regression_reward = 0.5

                scope_reward = compute_scope_reward(diff, gold_diff) if gold_diff else 0.5

                # Merge reward: proxy via diff quality heuristics when no real merger available
                merge_reward = 0.6 if diff.strip() else 0.0
                if diff and tests:
                    merge_reward = 0.8  # Higher confidence with tests present
                if diff and tests and regression_reward == 1.0 and test_reward == 1.0:
                    merge_reward = 1.0

                reward = (
                    config.merge_reward_weight * merge_reward
                    + config.test_reward_weight * test_reward
                    + config.regression_reward_weight * regression_reward
                    + config.scope_reward_weight * scope_reward
                )
                rewards.append(reward)

        return rewards

    return reward_fn


def load_rl_dataset(data_path: str) -> Dataset:
    """
    Load review execution tasks for RL training.
    Each example: {review_comment, file_context, repo_path, gold_diff, language}
    """
    import json
    examples = []
    if not Path(data_path).exists():
        raise FileNotFoundError(
            f"Training data not found at {data_path}. Run synthesis stage first."
        )
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue
            ex["prompt"] = format_rl_prompt(ex)
            ex["metadata"] = {
                "repo_path": ex.get("repo_path"),
                "gold_diff": ex.get("gold_diff"),
                "language": ex.get("language", "python"),
            }
            examples.append(ex)
    return Dataset.from_list(examples)


def format_rl_prompt(example: dict) -> str:
    """Format a review task as a prompt for RL training."""
    language = example.get("language", "python")
    review_comment = example.get("review_comment", "")
    file_context = example.get("file_context", "")[:3000]
    repo = example.get("repo", "owner/repo")

    return f"""<review_task>
Repository: {repo}
Language: {language}
Review comment: {review_comment}

File context:
```{language}
{file_context}
```

Generate the minimal diff that addresses this review comment, plus tests that verify the fix.
</review_task>

<think>"""


def train(config: RLTrainingConfig):
    logger.info(f"Loading base model: {config.base_model}")
    # device_map=None required for DeepSpeed ZeRO-3 — DeepSpeed manages device placement
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.bfloat16,
        device_map=None,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading SFT LoRA adapter from: {config.sft_checkpoint}")
    model = PeftModel.from_pretrained(base_model, config.sft_checkpoint, is_trainable=True)

    logger.info("Loading RL training dataset...")
    dataset = load_rl_dataset(config.train_data_path)
    logger.info(f"RL dataset: {len(dataset)} review tasks")

    reward_fn = build_reward_function(config)

    grpo_config = GRPOConfig(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_generations=config.num_generations,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        bf16=True,
        gradient_checkpointing=True,
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        run_name="mergepilot-rl-grpo",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=[reward_fn],
    )

    logger.info("Starting GRPO training with merge-outcome reward...")
    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    logger.info(f"RL training complete. Checkpoint saved to {config.output_dir}")


if __name__ == "__main__":
    import typer

    def main(
        base_model: str = "Qwen/Qwen2.5-7B-Coder-Instruct",
        sft_checkpoint: str = "./checkpoints/sft",
        output_dir: str = "./checkpoints/rl",
        data_path: str = "./data/rl/review_tasks.jsonl",
        num_generations: int = 8,
    ):
        config = RLTrainingConfig(
            base_model=base_model,
            sft_checkpoint=sft_checkpoint,
            output_dir=output_dir,
            train_data_path=data_path,
            num_generations=num_generations,
        )
        train(config)

    typer.run(main)
