# Absolutely! Fine-tuning a language model like LLaMA 3.2B involves several steps. Below is a concise example of how you might fine-tune the model using Python and the Hugging Face Transformers library. This example assumes you have a dataset ready for fine-tuning.

# Step 1: Install Required Libraries

# First, ensure you have the necessary libraries installed:

# pip install transformers datasets torch

# Step 2: Load the Model and Dataset

# Here's a basic script to load the model and dataset, and then fine-tune it:

from pathlib import Path
import torch
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B", token="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")
tokenizer.add_special_tokens
tokenizer.pad_token= "<|end_of_text|>"
tokenizer.padding_side="right"
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B", torch_dtype=torch.float32, device_map="auto")
tokenizer.pad_token= tokenizer.eos_token
# Load your dataset
dataset = load_dataset("text", data_files="C:\\Users\\Emilio Guillem\\Documents\\GIT\\IA\\src\\context_db\\context.txt", encoding='latin-1', split="train")
dataset.to_dict()
# dataset_train, dataset_eval =dataset.train_test_split(test_size=0.2, seed=42)
dataset = dataset.shuffle(seed=80)
# Tokenize the dataset
import nltk
nltk.download('punkt')
file_content = open("C:\\Users\\Emilio Guillem\\Documents\\GIT\\IA\\src\\context_db\\context.txt").read()
tokens_dt = nltk.word_tokenize(file_content)
def tokenize_function(examples):
    tokens_dt = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    tokens=[
        -100 if token == tokenizer.pad_token_id else token for token in tokens_dt
    ]
    return tokens

tokenized_datasets = dataset.map(tokenize_function)
tokenized_datasets = tokenized_datasets.remove_columns(['text'])
model.train()
# Set training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    eval_steps=40,
    logging_steps=40,
    save_steps=150,
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=2,
    fp16=False,
    log_level="info",
    weight_decay=0.01,
    max_grad_norm=2
)
# from trl import SFTConfig, SFTTrainer
# training_args = SFTConfig(
#     output_dir="./results",
#     optim="adamw_torch_fused",
#     evaluation_strategy="steps",
#     eval_steps=40,
#     logging_steps=10,
#     save_strategy="epoch",
#     learning_rate=2e-4,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     gradient_accumulation_steps=4,
#     gradient_checkpointing = True,
#     num_train_epochs=5,
#     fp16=False,
#     bf16=True,
#     log_level="info",
#     weight_decay=0.01,
#     max_grad_norm=0.3,
#     warmup_ratio=0.03,
#     lr_scheduler_type="cosine",
#     dataset_text_field="text",
#     report_to="tensorboard",
#     gradient_checkpointing_kwargs={"use_reentrant": False},
# )

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test']
)


# Fine-tune the model
trainer.train()

# Step 3: Save the Fine-Tuned Model

# After training, you can save the fine-tuned model:

# model.save_pretrained("C:/Users/Emilio Guillem/Documents/GIT/IA/src/llm/fine_tuned_llama")
# tokenizer.save_pretrained("C:/Users/Emilio Guillem/Documents/GIT/IA/src/llm/fine_tuned_llama")

# Notes:
# Dataset: Replace "path_to_your_dataset" with the actual path or name of your dataset.
# Hyperparameters: Adjust the hyperparameters (e.g., learning rate, batch size, number of epochs) as needed for your specific use case.
# Hardware: Ensure you have the necessary hardware (e.g., GPU) to handle the fine-tuning process, especially for large models like LLaMA 3.2B.

# Feel free to adapt this script to better fit your needs. Happy fine-tuning!
import os
os.environ['HF_TOKENIZER'] = "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"
model.push_to_hub("Emiliogs/orbital", use_auth_token=os.getenv("HF_TOKEN"))
tokenizer.push_to_hub("Emiliogs/orbital", use_auth_token=os.getenv("HF_TOKEN"))