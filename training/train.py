"""
Stage 1: Supervised Fine-Tuning (SFT)
Fine-tunes Qwen2.5-7B-Coder-Instruct on ~400k (review_comment, diff, tests) triples.
Uses LoRA rank 64 with DeepSpeed ZeRO-3 across 18× A6000.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import SFTConfig, SFTTrainer


@dataclass
class SFTTrainingConfig:
    base_model: str = "Qwen/Qwen2.5-7B-Coder-Instruct"
    output_dir: str = "./checkpoints/sft"

    # Training
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_seq_length: int = 16384

    # LoRA
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = None  # Auto-detected

    # Data paths (after synthesis)
    # Supports glob of data/training/*.jsonl or a single file
    review_pairs: str = "./data/training"
    dpo_pairs: str = "./data/synthesized/dpo_pairs.jsonl"

    # Logging
    logging_steps: int = 25
    save_steps: int = 500
    eval_steps: int = 500
    wandb_project: str = "mergepilot-sft"


SYSTEM_PROMPT = """\
You are MergePilot, an expert code reviewer that turns review comments into merged pull requests.
Given a review comment and repository context, generate the minimal diff that addresses the review
plus tests that prove the fix. Your PRs pass review on the first submission.

Always structure your response as:
<think>[Your analysis of what the review comment is asking for and the minimal change needed]</think>
<diff>[Unified diff format — only the lines that need to change, nothing more]</diff>
<tests>[New or modified tests that verify the fix]</tests>
"""


def build_training_messages(example: dict) -> list[dict]:
    """Build conversation messages for a synthesized review-to-PR pair."""
    repo = example.get("repo", "owner/repo")
    language = example.get("language", "python")
    review_comment = example.get("review_comment", "")
    file_context = example.get("file_context", "")
    reasoning = example.get("reasoning", "")
    diff = example.get("diff", "")
    tests = example.get("tests", "")

    user_msg = f"""Repository: {repo}
Language: {language}
Review comment: {review_comment}

File context:
```{language}
{file_context[:4000]}
```

Generate the minimal diff that addresses this review comment, plus tests that verify the fix."""

    assistant_msg = f"""<think>
{reasoning}
</think>

<diff>
{diff}
</diff>

<tests>
{tests}
</tests>"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]


def format_training_example(example: dict, tokenizer=None) -> str:
    """Format a synthesized review-to-PR pair into a training message.

    Uses tokenizer.apply_chat_template() when available to match inference format.
    Falls back to manual Qwen2.5 template if no tokenizer is provided.
    """
    messages = build_training_messages(example)
    if tokenizer is not None:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    # Fallback: manual Qwen2.5 chat template (kept for offline use)
    parts = []
    for msg in messages:
        parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
    return "\n".join(parts)


def load_all_training_data(config: SFTTrainingConfig, tokenizer=None) -> Dataset:
    """
    Load and combine all training streams from data/training/*.jsonl.

    Supports both a directory of JSONL files and a single JSONL file.
    """
    import json

    all_examples = []
    data_path = Path(config.review_pairs)

    if data_path.is_dir():
        # Load all *.jsonl files in the directory
        jsonl_files = sorted(data_path.glob("*.jsonl"))
        if not jsonl_files:
            raise FileNotFoundError(
                f"No *.jsonl files found in {data_path}. Run synthesis pipeline first."
            )
        for jsonl_file in jsonl_files:
            try:
                with open(jsonl_file) as f:
                    examples = [json.loads(line) for line in f if line.strip()]
                logger.info(f"Loaded {len(examples)} examples from {jsonl_file}")
                all_examples.extend(examples)
            except OSError as e:
                logger.warning(f"Skipping {jsonl_file}: {e}")
    elif data_path.is_file():
        try:
            with open(data_path) as f:
                all_examples = [json.loads(line) for line in f if line.strip()]
            logger.info(f"Loaded {len(all_examples)} examples from {data_path}")
        except OSError as e:
            logger.warning(f"Could not read {data_path}: {e}")
    else:
        logger.warning(f"Training data path not found: {data_path}")

    logger.info(f"Total training examples: {len(all_examples)}")

    formatted = [
        {"text": format_training_example(ex, tokenizer=tokenizer)}
        for ex in all_examples
    ]
    return Dataset.from_list(formatted)


def train(config: SFTTrainingConfig):
    logger.info(f"Loading base model: {config.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.bfloat16,
        use_cache=False,  # Required for gradient checkpointing
    )

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=config.lora_target_modules
        or [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    full_dataset = load_all_training_data(config, tokenizer=tokenizer)
    logger.info(f"Loaded {len(full_dataset)} total examples")
    split = full_dataset.train_test_split(test_size=0.02, seed=42)
    dataset = split["train"]
    eval_dataset = split["test"]
    logger.info(
        f"Training on {len(dataset)} examples, evaluating on {len(eval_dataset)} examples"
    )

    sft_config = SFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        bf16=True,
        gradient_checkpointing=True,
        deepspeed=str(Path(__file__).parent / "configs" / "ds_config.json"),
        report_to=[],
        run_name="mergepilot-sft",
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
        packing=False,
    )

    logger.info("Starting SFT training...")
    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    logger.info(f"SFT training complete. Model saved to {config.output_dir}")


if __name__ == "__main__":
    import typer

    def main(
        base_model: str = "Qwen/Qwen2.5-7B-Coder-Instruct",
        output_dir: str = "./checkpoints/sft",
        epochs: int = 3,
        data_path: str = "./data/synthesized/review_pairs.jsonl",
    ):
        config = SFTTrainingConfig(
            base_model=base_model,
            output_dir=output_dir,
            num_train_epochs=epochs,
            review_pairs=data_path,
        )
        train(config)

    typer.run(main)
