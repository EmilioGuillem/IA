#!/usr/bin/env python3
"""Fine-tune a local causal LM on conversation records.

Expect a local JSONL file with simple entries. Supported record shapes:
- {"text": "full conversation or prompt"}
- {"user": "...", "assistant": "..."}
- {"messages": [{"role":"user","content":"..."}, ...]}

This script tokenizes each example using the model tokenizer and trains the model.
It supports optional LoRA via `--use_lora` if `peft` is installed.
"""
from argparse import ArgumentParser
from pathlib import Path
import os

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from prompt_template import build_prompt if False else None
from conversation_logger import combine_conversations, normalize_conversations_for_training, archive_conversations

# local helper to avoid circular imports when used from package
def build_prompt_from_example(example_text: str, system_prompt: str = None) -> str:
    # Simple passthrough: user can craft full 'text' records.
    return example_text


def get_text_field(example):
    if 'text' in example:
        return example['text']
    if 'user' in example and 'assistant' in example:
        return example['user'] + '\n' + example['assistant']
    if 'messages' in example:
        out = []
        for m in example['messages']:
            role = m.get('role', '')
            content = m.get('content', m.get('text', ''))
            out.append(f"{role}: {content}")
        return '\n'.join(out)
    # fallback: stringify
    return str(example)


def tokenize_function(examples, tokenizer, max_length=2048):
    texts = [build_prompt_from_example(get_text_field(x)) for x in examples['text']]
    toks = tokenizer(texts, truncation=True, padding='longest', max_length=max_length)
    toks['labels'] = toks['input_ids'].copy()
    return toks


def main():
    p = ArgumentParser()
    p.add_argument('--model_path', required=True, help='Local base model directory')
    p.add_argument('--train_file', required=False, help='Local JSONL training file (if omitted will combine logs in python-model/data)')
    p.add_argument('--normalize', action='store_true', help='Normalize daily conversation files into prompt/response training file before training')
    p.add_argument('--archive_days', type=int, default=0, help='If >0, archive conversation files older than this number of days after combining')
    p.add_argument('--output_dir', required=True, help='Where to store the fine-tuned model')
    p.add_argument('--per_device_train_batch_size', type=int, default=1)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--max_length', type=int, default=2048)
    p.add_argument('--use_lora', action='store_true')
    args = p.parse_args()

    model_path = args.model_path
    train_file = args.train_file
    if not train_file:
        # combine daily conversation files into a single training file
        train_file = str(combine_conversations())
        print('No se proporcionó --train_file. Se combinaron los archivos diarios de conversaciones en', train_file)

    # optionally normalize into prompt/response format
    if args.normalize:
        norm_path = str(normalize_conversations_for_training())
        print('Se normalizó el histórico en formato prompt/response en', norm_path)
        train_file = norm_path

    # optionally archive old conversation files
    if args.archive_days and args.archive_days > 0:
        moved = archive_conversations(args.archive_days)
        print(f'Archivados {moved} archivos de conversaciones con más de {args.archive_days} días')
    output_dir = args.output_dir

    print('Loading tokenizer from', model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    # load dataset
    print('Loading dataset', train_file)
    ds = load_dataset('json', data_files={'train': train_file}, split='train')

    # ensure a 'text' column exists for our tokenizer helper
    def ensure_text(example):
        example['text'] = get_text_field(example)
        return example

    ds = ds.map(ensure_text)

    # tokenize
    tokenized = ds.map(lambda ex: tokenize_function({'text': [ex['text']]}, tokenizer, max_length=args.max_length), batched=False)

    # data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # model
    print('Loading model (this can use a lot of RAM)')
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

    # optional LoRA (if peft available)
    if args.use_lora:
        try:
            from peft import get_peft_model, LoraConfig, TaskType
            from peft import prepare_model_for_kbit_training
            # attempt to prepare
            try:
                model = prepare_model_for_kbit_training(model)
            except Exception:
                pass
            lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=['q_proj', 'v_proj'],
                lora_dropout=0.1,
                bias='none',
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lora_config)
            print('Applied LoRA adapters')
        except Exception as e:
            print('Could not enable LoRA (peft missing or incompatible):', e)
            print('Continuing without LoRA')

    # training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_strategy='epoch',
        fp16=True if os.getenv('USE_FP16', '1') == '1' else False,
        push_to_hub=False,
        remove_unused_columns=False,
        gradient_accumulation_steps=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print('Starting training...')
    trainer.train()
    print('Saving model to', output_dir)
    trainer.save_model(output_dir)


if __name__ == '__main__':
    main()
