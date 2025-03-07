# from numba import jit, cuda
import numpy as np
from datasets import load_dataset
import os
import torch
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling

def main():

    # Check if CUDA is available
    print(torch.cuda.is_available()) # True if CUDA is available

    # Get the number of GPUs available
    print(torch.cuda.device_count()) # Number of GPUs

    # Get the name of the current GPU
    print(torch.cuda.get_device_name(0)) #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()



    dataset_train, dataset_eval = load_dataset("json", data_files="C:\\Users\\Emilio\\Documents\\GitHub\\IA\\src\\context_db\\context.json", split=['train[:80%]', 'train[80%:]'])


    # tokenizer = AutoTokenizer.from_pretrained("./orbital_llama32_1B/orbital", token="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")
    tokenizer = AutoTokenizer.from_pretrained('C:\\Users\\Emilio\\Documents\\GitHub\\IA\\orbital_llama32_1B', token="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")

    # while(True):
    # model = AutoModelForCausalLM.from_pretrained("./orbital_llama32_1B/orbital", token="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", low_cpu_mem_usage=True, 
    #                                              torch_dtype=torch.float16, device_map='auto')
    model = AutoModelForCausalLM.from_pretrained('C:\\Users\\Emilio\\Documents\\GitHub\\IA\\orbital_llama32_1B', token="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", low_cpu_mem_usage=True, 
                                                torch_dtype=torch.float16, device_map='auto') 
    
    # "meta-llama/Llama-3.2-1B"
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    def tokenize_function(examples):
        tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        #new
        tokenizer.padding_side="right"
        if isinstance(examples["content"], list):
            examples["content"] = [str(text) for text in examples["content"]]
        else:
            examples["content"] = str(examples["content"])
            
        return tokenizer(examples['content'],  padding='max_length', truncation=True, max_length=512)


    train_tokenized_datasets = dataset_train.map(tokenize_function, batched=True)
    eval_tokenized_datasets = dataset_eval.map(tokenize_function, batched=True)

    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, mlm_probability=0.15)


    training_args= TrainingArguments(
        output_dir="./orbital_llama32_1B",
        eval_strategy="steps",
        eval_steps=10,
        logging_steps=10,
        save_steps=40,
        learning_rate=2e-5,
        # per_device_train_batch_size=2,
        # per_device_eval_batch_size=2,
        auto_find_batch_size=True,
        num_train_epochs=1,
        fp16=False,
        log_level="info",
        # weight_decay=0.01,
        # max_grad_norm=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized_datasets,
        eval_dataset=eval_tokenized_datasets,
        data_collator=data_collator,
    )
        
    #send to GPU
    
    # with torch.no_grad():
    torch.inference_mode()
    torch.cuda.empty_cache()
    state  = model.state_dict()
    torch.save(state, './orbital_llama32_1B/orbital')
    # move the model parameter to cpu
    state = torch.load('./orbital_llama32_1B/orbital', map_location=torch.device('cpu'))

    model.load_state_dict(state)

    # now move the model parameter to a GPU device of your choice
    model.to(device)
    torch.cuda.empty_cache()
    
    trainer.train()

    #save model
    # torch.save(state, './orbital_llama32_1B/orbital')
    model.save_pretrained('C:\\Users\\Emilio\\Documents\\GitHub\\IA\\orbital_llama32_1B')
    tokenizer.save_pretrained('C:\\Users\\Emilio\\Documents\\GitHub\\IA\\orbital_llama32_1B')
    

if __name__ == "__main__":
    main()