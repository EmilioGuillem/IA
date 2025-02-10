from datasets import load_dataset

# dataset = load_dataset("json", data_files="C:\\Users\\Emilio Guillem\\Documents\\GIT\\IA\\src\\context_db\\context.json", split='train')
# dataset = load_dataset("json", data_files="C:\\Users\\Emilio Guillem\\Documents\\GIT\\IA\\src\\context_db\\context.json", encoding='latin-1', split="train")
# dataset_train = load_dataset("json", data_files="C:\\Users\\Emilio Guillem\\Documents\\GIT\\IA\\src\\context_db\\context.json", split=['train[:80%]', 'train[80%:]'])
# dataset_eval = load_dataset("json", data_files="C:\\Users\\Emilio Guillem\\Documents\\GIT\\IA\\src\\context_db\\context.json", split=['train[:20%]', 'train[20%:]'])
dataset_train, dataset_eval = load_dataset("json", data_files="C:\\Users\\Emilio Guillem\\Documents\\GIT\\IA\\src\\context_db\\context.json", split=['train[:80%]', 'train[80%:]'])

from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")

def tokenize_function(examples):
    tokenizer.pad_token = tokenizer.eos_token
    if isinstance(examples["content"], list):
        examples["content"] = [str(text) for text in examples["content"]]
    else:
        examples["content"] = str(examples["content"])
        
    return tokenizer(examples['content'], padding="max_length", truncation=True)


train_tokenized_datasets = dataset_train.map(tokenize_function, batched=True)
eval_tokenized_datasets = dataset_eval.map(tokenize_function, batched=True)
# Split the dataset
# train_data, eval_data = train_test_split(tokenized_datasets, test_size=0.2, random_state=42)

# small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
# small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# small_train_dataset = tokenized_datasets.shuffle(seed=42).select(range(1000))
# small_eval_dataset = tokenized_datasets.shuffle(seed=42).select(range(1000))

#trainin pytorch model
# from transformers import AutoModelForSequenceClassification

# model = AutoModelForSequenceClassification.from_pretrained("meta-llama/Llama-3.2-3B", num_labels=5, torch_dtype="auto")
# from transformers import TrainingArguments

# training_args = TrainingArguments(output_dir="test_trainer")

# import numpy as np
# import evaluate

# metric = evaluate.load("accuracy")

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)

# from transformers import TrainingArguments, Trainer

# training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch")

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=small_train_dataset,
#     eval_dataset=small_eval_dataset,
#     compute_metrics=compute_metrics,
# )
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B", device_map="auto")

#training model
from transformers import Trainer, TrainingArguments
# import torch
# use_gpu = torch.cuda.is_available()
# print(use_gpu)

# model.cuda()
import tensorflow as tf
# Check if GPU is available
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())



training_args= TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=8,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized_datasets,
    eval_dataset=eval_tokenized_datasets,
)

with tf.device('/GPU:0'):
    trainer.train()

#save model
model.save_pretrained('./fine-tuned-llama3')
tokenizer.save_pretrained('./fine-tuned-llama3')
#keras(https://huggingface.co/docs/transformers/training)