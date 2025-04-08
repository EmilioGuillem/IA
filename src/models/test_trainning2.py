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
from peft import PeftConfig, PeftModel, AutoPeftModelForCausalLM
from peft import LoraConfig
import datetime

def main():
    # path_to_save_model = 'C:\\Users\\Emilio\\Documents\\GitHub\\IA\\src\\llm\\llama31_orbital_chat_8B_q4'
    # path_to_model = 'meta-llama/Llama-3.3-70B-Instruct'
    # path_to_save_model = 'C:\\Users\\Emilio\\Documents\\GitHub\\IA\\src\\llm\\llama31_orbital_chat_8B_q4_v1'
    # path_to_model = 'meta-llama/Llama-3.1-8B-Instruct'
    path_to_save_model = 'C:\\Users\\Emilio\\Documents\\GitHub\\IA\\src\\llm\\llama32_orbital_chat_3B_q4'
    path_to_model = 'meta-llama/Llama-3.2-3B-Instruct'
    
    #----------------LLAMA 4 -------------------------------------------
    # path_to_save_model = 'C:\\Users\\Emilio\\Documents\\GitHub\\IA\\src\\llm\\llama4_orbital_chat_17B_128E_q4'
    # path_to_model = 'meta-llama/Llama-4-Maverick-17B-128E-Instruct'
    
    # path_to_save_model = 'C:\\Users\\Emilio\\Documents\\GitHub\\IA\\src\\llm\\llama4_orbital_chat_17B_16E_q4'
    # path_to_model = 'meta-llama/Llama-4-Scout-17B-16E-Instruct'
    
     # path_to_save_model = 'C:\\Users\\Emilio\\Documents\\GitHub\\IA\\src\\llm\\llama4_orbital_chat_17B_128E_FP8_q4'
    # path_to_model = 'meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8'
    
    
    #MODELOS QUANTIZADOS
    # path_to_save_model = 'C:\\Users\\Emilio\\Documents\\GitHub\\IA\\src\\llm\\llama4_orbital_chat_17B_16E_8bit_q4'
    # path_to_model = 'unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-bnb-8bit'
    
    # path_to_save_model = 'C:\\Users\\Emilio\\Documents\\GitHub\\IA\\src\\llm\\llama4_orbital_chat_17B_16E_4bit_q4'
    # path_to_model = 'unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-bnb-4bit'
    
    # path_to_save_model = 'C:\\Users\\Emilio\\Documents\\GitHub\\IA\\src\\llm\\llama4_orbital_chat_17B_16E_q4_Instruct'
    # path_to_model = 'unsloth/Llama-4-Scout-17B-16E-Instruct'
    
    # path_to_save_model = 'C:\\Users\\Emilio\\Documents\\GitHub\\IA\\src\\llm\\llama4_orbital_chat_17B_128E_q4_Instruct'
    # path_to_model = 'unsloth/Llama-4-Maverick-17B-128E-Instruct'
    
    # path_to_save_model = 'C:\\Users\\Emilio\\Documents\\GitHub\\IA\\src\\llm\\llama4_orbital_chat_17B_128E_q4_Instruct_FP8'
    # path_to_model = 'unsloth/Llama-4-Maverick-17B-128E-Instruct-FP8'
    
    # path_to_save_model = 'C:\\Users\\Emilio\\Documents\\GitHub\\IA\\src\\llm\\llama4_orbital_chat_17B_16E_q4_4bit'
    # path_to_model = 'mlx-community/Llama-4-Maverick-17B-16E-Instruct-4bit'
    
    # path_to_save_model = 'C:\\Users\\Emilio\\Documents\\GitHub\\IA\\src\\llm\\llama4_orbital_chat_17B_16E_q4_6bit'
    # path_to_model = 'mlx-community/Llama-4-Maverick-17B-16E-Instruct-6bit'
    
    #download for ollama
    # lmstudio-community/Llama-4-Scout-17B-16E-Instruct-GGUF
    
    #--------------NO FUNCIONAN, POCA RAM EN CPU-------------------------
    # path_to_model = 'meta-llama/Llama-3.3-70B-Instruct-evals'
    # path_to_model = 'meta-llama/Llama-3.2-11B-Vision-Instruct'
    #--------------------------------------------------------------------
    
    # minlik/docllm-yi-34b #document understanding
    # JinghuiLuAstronaut/DocLLM_baichuan2_7b
    # SantiagoPG/DOC_QA
    # Check if CUDA is available
    print(torch.cuda.is_available()) # True if CUDA is available

    # Get the number of GPUs available
    print(torch.cuda.device_count()) # Number of GPUs

    # Get the name of the current GPU
    print(torch.cuda.get_device_name(0)) #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

#     LLAMA_3_CHAT_TEMPLATE = (
#     "{% for message in messages %}"
#         "{% if message['role'] == 'system' %}"
#             "{{ message['content'] }}"
#         "{% elif message['role'] == 'user' %}"
#             "{{ '\n\nUser: ' + message['content'] +  eos_token }}"
#         "{% elif message['role'] == 'assistant' %}"
#             "{{ '\n\nAssistant: '  + message['content'] +  eos_token  }}"
#         "{% endif %}"
#     "{% endfor %}"
#     "{% if add_generation_prompt %}"
#     "{{ '\n\nAssistant: ' }}"
#     "{% endif %}"
# )
    now = str(datetime.datetime.now().today().strftime("%d-%m-%Y %H:%M:%S"))
    LLAMA_3_CHAT_TEMPLATE= (
    {
      "role": "system",
      "content": "Tu nombre es Orbital, un asistente virtual",
      "timestamp": "{now}"
    },
    {
      "role": "user",
      "content": "Hola, Orbital",
      "timestamp": "{now}"
    },
    {
      "role": "assistant",
      "content": "Hola, mi nombre es Orbital, un asistente virtual listo para ayudarte.",
      "timestamp": "{now}"
    }
    )


    dataset_train, dataset_eval = load_dataset("json", data_files="C:\\Users\\Emilio\\Documents\\GitHub\\IA\\src\\context_db\\context_v1.json",encoding='latin1',  split=['train[:80%]', 'train[80%:]'])
    

    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", token="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")
    tokenizer = AutoTokenizer.from_pretrained(path_to_model, token="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")
    tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE
    
    # Model    
    torch_dtype = torch.bfloat16
    quant_storage_dtype = torch.bfloat16
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
    
    quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_storage=quant_storage_dtype,
        )
    
    model = AutoModelForCausalLM.from_pretrained(
        path_to_model,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        device_map='auto',
        token="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",
        attn_implementation="sdpa", # use sdpa, alternatively use "flash_attention_2"
        torch_dtype=quant_storage_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,  # this is needed for gradient checkpointing
    )
    
    
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
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
            
        return tokenizer(examples['content'], padding='max_length', truncation=True, max_length=512)


    train_tokenized_datasets = dataset_train.map(tokenize_function, batched=True)
    eval_tokenized_datasets = dataset_eval.map(tokenize_function, batched=True)

    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, mlm_probability=0.15)
    ################
    # PEFT
    ################

    # LoRA config based on QLoRA paper & Sebastian Raschka experiment
    peft_config = LoraConfig(
        lora_alpha=8,
        lora_dropout=0.05,
        r=16,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        modules_to_save = ["lm_head", "embed_tokens"] # add if you want to use the Llama 3 instruct template
    )
    
    model.add_adapter(peft_config)

    # model = AutoPeftModelForCausalLM.from_pretrained(model.state_dict(), 
    #                                                  low_cpu_mem_usage = True, torch_dtype=torch.bfloat16, peft_config=peft_config, device = 'auto')
    # model.merge_and_unload()
    ################
    # Training
    ################
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized_datasets,
        eval_dataset=eval_tokenized_datasets,
        peft_config=peft_config,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    if trainer.accelerator.is_main_process:
        trainer.model.print_trainable_parameters()

    ##########################
    # Train model
    ##########################
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)

    ##########################
    # SAVE MODEL FOR SAGEMAKER
    ##########################
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()
    
    #send to GPU
    
    # # with torch.no_grad():
    # torch.inference_mode()
    # torch.cuda.empty_cache()
    state  = model.state_dict()
    torch.save(state, path_to_save_model+'\\orbital')
    # move the model parameter to cpu
    state = torch.load(path_to_save_model+'\\orbital', map_location=torch.device('cpu'))

    # # torch.save(state, path_to_save_model+'\\audio_classifier')
    # # # move the model parameter to cpu
    # # state = torch.load(path_to_save_model+'\\audio_classifier', map_location=torch.device('cpu'))

    # # torch.save(state, path_to_save_model+'\\document_answering')
    # # # move the model parameter to cpu
    # # state = torch.load(path_to_save_model+'\\document_answering', map_location=torch.device('cpu'))


    # #save model
    # # torch.save(state, './TrainingTest/orbital')
    # model = AutoPeftModelForCausalLM.from_pretrained(path_to_save_model, low_cpu_mem_usage = True, torch_dtype=torch.bfloat16, peft_config=peft_config)
    # model.merge_and_unload()
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
    os.environ['HF_TOKENIZER'] = "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"
    model.push_to_hub("emiliogsAI/test2", use_auth_token=os.getenv("HF_TOKENIZER"))
    
    tokenizer.push_to_hub("emiliogsAI/test2", use_auth_token=os.getenv("HF_TOKENIZER"))

    # model.save_pretrained(path_to_save_model, safe_serialization=True, max_shard_size='3GB')
    # tokenizer.save_pretrained(path_to_save_model)

if __name__ == "__main__":
    main()