# This file has been created (totally or partially) with the assistance of artificial intelligence tools.
# All content has been generated under the direct supervision of a named individual,
# and under the AI.Backbone Orchestrator Compliance framework

"""Fine-tune a causal language model with optional Hugging Face dependencies."""

from __future__ import annotations

import argparse

from code_agent.prompts import build_chat_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a local causal language model.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-train-epochs", type=int, default=3)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=1024)
    return parser.parse_args()


def make_prompt(example: dict[str, str]) -> str:
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    target = example.get("output", "")
    if input_text:
        user = f"Instruction: {instruction}\nInput: {input_text}"
    else:
        user = f"Instruction: {instruction}"
    return build_chat_prompt(user) + target


def main() -> int:
    args = parse_args()
    try:
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
    except ModuleNotFoundError as exc:
        raise SystemExit("Install optional dependencies first: python -m pip install -e .[hf]") from exc

    dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    def tokenize_batch(examples: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        rows = [dict(zip(examples, values)) for values in zip(*examples.values())]
        texts = [make_prompt(row) for row in rows]
        outputs = tokenizer(texts, truncation=True, max_length=args.max_length, padding="longest")
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs

    tokenized_dataset = dataset.map(tokenize_batch, batched=True, remove_columns=dataset.column_names)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        fp16=False,
        push_to_hub=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())