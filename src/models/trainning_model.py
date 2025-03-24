# from numba import jit, cuda
import numpy as np
from datasets import load_dataset
import os
import torch
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers import AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling
from trl import SFTTrainer
from peft import PeftConfig, PeftModel

def main():
    path_to_save_model = 'C:\\Users\\Emilio\\Documents\\GitHub\\IA\\src\\llm\\llama32_orbital_chat_3B'
    path_to_hggf_model = 'C:\\Users\\Emilio\\Documents\\GitHub\\IA\\src\\llm\\llama32_orbital_chat_3B'
    # Check if CUDA is available
    print(torch.cuda.is_available()) # True if CUDA is available

    # Get the number of GPUs available
    print(torch.cuda.device_count()) # Number of GPUs

    # Get the name of the current GPU
    print(torch.cuda.get_device_name(0)) #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )


    dataset_train, dataset_eval = load_dataset("json", data_files="C:\\Users\\Emilio\\Documents\\GitHub\\IA\\src\\context_db\\context.json",encoding='latin1',  split=['train[:80%]', 'train[80%:]'])
    

    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", token="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")
    tokenizer = AutoTokenizer.from_pretrained(path_to_hggf_model, token="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")

    # while(True):
    # model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", token="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", low_cpu_mem_usage=True, 
                                                #  torch_dtype=torch.float16, device_map='auto')
    model = AutoModelForCausalLM.from_pretrained(path_to_hggf_model, token="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", low_cpu_mem_usage=True, 
                                                torch_dtype=torch.bfloat16, device_map='auto') 
    # model = PeftModel.from_pretrained(model, )
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
        # return tokenizer(examples['content'],  padding='max_length', truncation=True)


    train_tokenized_datasets = dataset_train.map(tokenize_function, batched=True)
    eval_tokenized_datasets = dataset_eval.map(tokenize_function, batched=True)

    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, mlm_probability=0.15)


    training_args= TrainingArguments(
        output_dir=path_to_save_model,
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
        bf16=True,
        log_level="info",
        weight_decay=0.01,
        max_grad_norm=2,
        optim = "adamw_8bit",
    )

    trainer = SFTTrainer(
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
    # torch.save(state, path_to_save_model+'\\orbital')
    # # move the model parameter to cpu
    # state = torch.load(path_to_save_model+'\\orbital', map_location=torch.device('cpu'))

    # torch.save(state, path_to_save_model+'\\audio_classifier')
    # # move the model parameter to cpu
    # state = torch.load(path_to_save_model+'\\audio_classifier', map_location=torch.device('cpu'))

    # torch.save(state, path_to_save_model+'\\document_answering')
    # # move the model parameter to cpu
    # state = torch.load(path_to_save_model+'\\document_answering', map_location=torch.device('cpu'))

    model.load_state_dict(state)

    # now move the model parameter to a GPU device of your choice
    model.to(device)
    torch.cuda.empty_cache()
    
    trainer.train()

    #save model
    # torch.save(state, './TrainingTest/orbital')
    model.save_pretrained(path_to_save_model)
    tokenizer.save_pretrained(path_to_save_model)

    #save gguf for ollama serve
    # Step 3: Save the Fine-Tuned Model

    # After training, you can save the fine-tuned model:

    # model.save_pretrained("C:/Users/Emilio Guillem/Documents/GIT/IA/src/llm/fine_tuned_llama")
    # tokenizer.save_pretrained("C:/Users/Emilio Guillem/Documents/GIT/IA/src/llm/fine_tuned_llama")

    # Notes:
    # Dataset: Replace "path_to_your_dataset" with the actual path or name of your dataset.
    # Hyperparameters: Adjust the hyperparameters (e.g., learning rate, batch size, number of epochs) as needed for your specific use case.
    # Hardware: Ensure you have the necessary hardware (e.g., GPU) to handle the fine-tuning process, especially for large models like LLaMA 3.2B.

    # Feel free to adapt this script to better fit your needs. Happy fine-tuning!
    # import os
    # os.environ['HF_TOKENIZER'] = "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"
    # model.push_to_hub("Emiliogs/orbital", use_auth_token=os.getenv("HF_TOKEN"))
    # tokenizer.push_to_hub("Emiliogs/orbital", use_auth_token=os.getenv("HF_TOKEN"))


if __name__ == "__main__":
    main()