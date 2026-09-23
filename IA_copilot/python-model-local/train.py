# This file has been created (totally or partially) with the assistance of artificial intelligence tools.
# All content has been generated under the direct supervision of a named individual,
# and under the AI.Backbone Orchestrator Compliance framework

"""Fine-tune a local Hugging Face model and save reusable artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from code_agent.prompts import build_chat_prompt
from data_pipeline import load_records, write_prepared_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a local causal language model.")
    parser.add_argument("--model-name-or-path", required=True, help="Hugging Face base model or local Transformers model")
    parser.add_argument("--data", nargs="+", required=True, help="Conversation JSONL/JSON, text files, or directories")
    parser.add_argument("--output-dir", required=True, help="Directory for the trained model or LoRA adapter")
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--use-lora", action="store_true", help="Train small LoRA adapters instead of all weights")
    return parser.parse_args()


def make_prompt(text: str) -> str:
    return build_chat_prompt(text)


def main() -> int:
    args = parse_args()
    records = load_records(args.data)
    output_dir = Path(args.output_dir)
    prepared_path = write_prepared_dataset(records, output_dir / "prepared_dataset.jsonl")

    try:
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
    except ModuleNotFoundError as exc:
        raise SystemExit("Install training dependencies first: python -m pip install -e .[hf,training]") from exc

    dataset = load_dataset("json", data_files=str(prepared_path), split="train")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )

    if args.use_lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ModuleNotFoundError as exc:
            raise SystemExit("Install LoRA dependencies first: python -m pip install -e .[training]") from exc
        model = get_peft_model(
            model,
            LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            ),
        )

    def tokenize_batch(examples: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        texts = [make_prompt(text) for text in examples["text"]]
        outputs = tokenizer(texts, truncation=True, max_length=args.max_length, padding="longest")
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs

    tokenized = dataset.map(tokenize_batch, batched=True, remove_columns=dataset.column_names)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=False,
        report_to="none",
        push_to_hub=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Training artifacts saved to {output_dir}")
    print(f"Prepared dataset saved to {prepared_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())