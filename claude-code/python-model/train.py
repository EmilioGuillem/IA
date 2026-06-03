import argparse
import os
from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from prompt_template import build_chat_prompt


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune a causal language model.')
    parser.add_argument('--model_name_or_path', required=True)
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--per_device_train_batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--max_length', type=int, default=1024)
    return parser.parse_args()


def make_prompt(example):
    instruction = example.get('instruction', '')
    input_text = example.get('input', '')
    target = example.get('output', '')

    if input_text:
        user = f"Instruction: {instruction}\nInput: {input_text}"
    else:
        user = f"Instruction: {instruction}"

    prompt = build_chat_prompt(user)
    prompt += target
    return prompt


def tokenize_fn(examples, tokenizer, max_length):
    texts = [make_prompt(example) for example in examples]
    outputs = tokenizer(texts, truncation=True, max_length=max_length, padding='longest')
    outputs['labels'] = outputs['input_ids'].copy()
    return outputs


def main():
    args = parse_args()
    dataset = load_dataset('json', data_files=args.dataset_path, split='train')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    tokenized_dataset = dataset.map(
        lambda examples: tokenize_fn(examples, tokenizer, args.max_length),
        batched=True,
        remove_columns=dataset.column_names,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

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
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == '__main__':
    main()
