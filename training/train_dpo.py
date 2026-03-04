"""
Stage 3 — DPO Training

Direct Preference Optimization using:
  - Chosen: high-impact, specific, actionable reviews
  - Rejected: vague, low-quality, or incorrect reviews

Run after Stage 2 (GRPO RL). Produces final model checkpoint.

Usage:
  python -m training.train_dpo --base-model ./checkpoints/stage2 --output ./checkpoints/stage3-dpo
"""

import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import Dataset
from loguru import logger
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import DPOConfig, DPOTrainer


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DPODataset:
    """Load and prepare DPO preference pairs for training."""

    def __init__(
        self,
        data_dir: str | Path,
        tokenizer,
        max_length: int = 2048,
        max_pairs: int = 30_000,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_pairs = max_pairs

    def load(self) -> Dataset:
        pairs = []

        # Load synthesized DPO pairs
        dpo_file = self.data_dir / "dpo_pairs.jsonl"
        if dpo_file.exists():
            with open(dpo_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        pair = self._normalize(item)
                        if pair:
                            pairs.append(pair)
                    except (json.JSONDecodeError, KeyError):
                        continue

        # Load from evaluation-judged pairs (high vs low quality)
        judged_file = self.data_dir / "judged_pairs.jsonl"
        if judged_file.exists():
            with open(judged_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        pair = self._normalize_judged(item)
                        if pair:
                            pairs.append(pair)
                    except (json.JSONDecodeError, KeyError):
                        continue

        if not pairs:
            logger.warning("No DPO pairs found — using synthetic fallback")
            pairs = self._create_synthetic_pairs()

        pairs = pairs[: self.max_pairs]
        logger.info(f"DPO dataset: {len(pairs):,} preference pairs")

        return Dataset.from_list(pairs)

    def _normalize(self, item: dict) -> Optional[dict]:
        """Normalize a synthesized DPO pair."""
        chosen = item.get("chosen")
        rejected = item.get("rejected")
        prompt = item.get("prompt", "")

        if not chosen or not rejected or not prompt:
            return None

        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }

    def _normalize_judged(self, item: dict) -> Optional[dict]:
        """Normalize a judge-evaluated pair."""
        diff = item.get("diff", "")
        review_a = item.get("review_a", "")
        review_b = item.get("review_b", "")
        chosen_label = item.get("chosen", "A")

        if not diff or not review_a or not review_b:
            return None

        chosen = review_a if chosen_label == "A" else review_b
        rejected = review_b if chosen_label == "A" else review_a

        prompt = f"Please review this pull request:\n\n```diff\n{diff[:2000]}\n```"

        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }

    def _create_synthetic_pairs(self) -> list[dict]:
        """Synthetic fallback pairs for testing."""
        synthetic_example = {
            "prompt": "Please review this code:\n```diff\n+def get_user(id):\n+    return db.query(f'SELECT * FROM users WHERE id={id}')\n```",
            "chosen": (
                "[SECURITY] File: api/users.py, Line: 2\n"
                "Observation: SQL query built with f-string interpolation.\n"
                "Issue: This allows SQL injection — an attacker can pass `1 OR 1=1` as id.\n"
                "Suggestion: Use parameterized queries.\n"
                "Example:\n```python\ndb.query('SELECT * FROM users WHERE id = %s', (id,))\n```"
            ),
            "rejected": "This might have some issues. Consider refactoring this function.",
        }
        # Use dict() to create independent copies — list multiplication creates shared references
        return [dict(synthetic_example) for _ in range(10)]


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class DPOMetricsCallback(TrainerCallback):
    """Log DPO-specific metrics during training."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            rewards_chosen = logs.get("rewards/chosen", None)
            rewards_rejected = logs.get("rewards/rejected", None)
            if rewards_chosen is not None and rewards_rejected is not None:
                margin = rewards_chosen - rewards_rejected
                logger.info(
                    f"Step {state.global_step}: "
                    f"reward_margin={margin:.4f}, "
                    f"chosen={rewards_chosen:.4f}, "
                    f"rejected={rewards_rejected:.4f}"
                )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_dpo(
    base_model: str,
    data_dir: str,
    output_dir: str,
    batch_size: int = 4,
    grad_accum: int = 4,
    learning_rate: float = 5e-7,
    beta: float = 0.1,
    max_steps: int = 2000,
    max_length: int = 2048,
    lora_rank: int = 32,
) -> None:
    """
    Stage 3: DPO fine-tuning.

    Args:
        base_model: Path to Stage 2 GRPO checkpoint (or Stage 1 SFT)
        data_dir: Directory with dpo_pairs.jsonl (and optionally judged_pairs.jsonl)
        output_dir: Where to save Stage 3 checkpoint
        beta: DPO temperature (0.1 = standard, higher = more conservative)
    """
    logger.info(f"Stage 3 DPO training: {base_model} -> {output_dir}")
    logger.info(f"beta={beta}, lr={learning_rate}, steps={max_steps}")

    # Load tokenizer and model
    # base_model may point to a PEFT adapter checkpoint (Stage 2 GRPO output).
    # Load the foundation model first, then overlay the PEFT adapter.
    BASE_FOUNDATION_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Coder-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_FOUNDATION_MODEL,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False,
    )
    model = PeftModel.from_pretrained(base, base_model)  # base_model = PEFT adapter dir
    model = model.merge_and_unload()  # merge for DPO stability
    model.train()

    # Reference model (frozen copy for KL)
    base_ref = AutoModelForCausalLM.from_pretrained(
        BASE_FOUNDATION_MODEL,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model_ref = PeftModel.from_pretrained(base_ref, base_model)
    model_ref = model_ref.merge_and_unload()
    model_ref.eval()
    for p in model_ref.parameters():
        p.requires_grad = False

    # Apply LoRA
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # Load dataset
    dataset = DPODataset(data_dir, tokenizer, max_length=max_length).load()
    split = dataset.train_test_split(test_size=0.02, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    logger.info(f"Train: {len(train_dataset):,} pairs, Eval: {len(eval_dataset):,} pairs")

    # DPO config
    dpo_config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        beta=beta,
        max_length=max_length,
        max_prompt_length=max_length // 2,
        bf16=True,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=500,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to=[],
        run_name="mergepilot-dpo",
        deepspeed=str(Path(__file__).parent / "configs" / "ds_config.json"),
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=model_ref,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[DPOMetricsCallback()],
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Merge LoRA weights — calling merge_and_unload() directly on the trainer's
    # model is unsafe under ZeRO-3 (weights may still be sharded across devices).
    # Instead reload on CPU from the saved checkpoint and then merge.
    logger.info("Merging LoRA adapter weights...")
    merged_output = os.path.join(output_dir, "merged")
    _base_for_merge = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
    )
    _peft_for_merge = PeftModel.from_pretrained(_base_for_merge, output_dir)
    merged_model = _peft_for_merge.merge_and_unload()
    merged_model.save_pretrained(merged_output)
    tokenizer.save_pretrained(merged_output)

    logger.success(f"DPO training complete. Final model: {merged_output}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="./checkpoints/stage2")
    parser.add_argument("--data-dir", default="./data/synthesized")
    parser.add_argument("--output-dir", default="./checkpoints/stage3-dpo")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--lora-rank", type=int, default=32)
    args = parser.parse_args()

    train_dpo(
        base_model=args.base_model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        max_steps=args.max_steps,
        lora_rank=args.lora_rank,
    )
