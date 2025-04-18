# from numba import jit, cuda
import numpy as np
from datasets import load_dataset
import os


import torch
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoConfig
from transformers import AutoModelForCausalLM, AutoModel
from transformers import DataCollatorForLanguageModeling
from trl import SFTTrainer
from peft import PeftConfig, PeftModel, AutoPeftModelForCausalLM
from peft import LoraConfig

import LlamaCPP as llama

import datetime

def main():

    # #---------------------ACTUAL------------------------------------------------
    path_to_save_model = 'C:\\Users\\Emilio\\Documents\\GitHub\\IA\\src\\llm\\test\\'
    path_to_save_file = 'Llama-orbital-3.2-3B-Instruct-Q4_K_M.gguf'
    path_to_model = 'C:\\Users\\Emilio\\Documents\\GitHub\\IA\\src\\llm\\'
    gguf_file_model = 'Llama-3.2-3B-Instruct-Q4_K_M.gguf'
    path_to_config_model = 'meta-llama/Llama-3.2-3B-Instruct'

    # Check if CUDA is available
    print(torch.cuda.is_available()) # True if CUDA is available

    # Get the number of GPUs available
    print(torch.cuda.device_count()) # Number of GPUs

    # Get the name of the current GPU
    print(torch.cuda.get_device_name(0)) #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    now = str(datetime.datetime.now().today().strftime("%d-%m-%Y %H:%M:%S"))


    dataset_train, dataset_eval = load_dataset("json", data_files="C:\\Users\\Emilio\\Documents\\GitHub\\IA\\src\\context_db\\context_v1.json",encoding='latin1',  split=['train[:80%]', 'train[80%:]'])

    torch.cuda.empty_cache()
    # Model    
    torch_dtype = torch.bfloat16
    quant_storage_dtype = torch.bfloat16
    model_config = AutoConfig.from_pretrained(path_to_config_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(path_to_model, gguf_file=gguf_file_model, low_cpu_mem_usage=True,
        device_map='auto',torch_dtype=quant_storage_dtype, config=model_config)
    #  = Llama.from_pretrained(filename='granite-3.2-8b-instruct-Q4_K_M.gguf', local_dir='C:\\Users\\Emilio\\Documents\\GitHub\\IA\\src\\llm\\test')
        

    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", token="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")
    tokenizer = AutoTokenizer.from_pretrained(path_to_config_model)
    
    
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

    #GUARDADO USO POSTERIOR
    
    
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
    ################
    # Training
    ################
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized_datasets,
        eval_dataset=eval_tokenized_datasets,
        # peft_config=peft_config,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

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
    trainer.save_model(path_to_save_model+'\\output')

    # -------------------------------------LORA--------------------------------------

    model = PeftModel.from_pretrained(model, path_to_save_model+'\\output')
    # model = model.merge_and_unload()
    # model.save_pretrained("modelo_completo")


    # ------------------------------------------SAVE MODEL-----------------------------------------------------------------
    # 🔥 Fusiona solo si los adaptadores están activos
    if hasattr(model, 'peft_config') and model.active_adapter:
        model = model.merge_and_unload()
        try:
            model.save_pretrained(path_to_save_model+'\\output')
        except:
            CONFIG_NAME = "config.json"
            WEIGHTS_NAME = "orbital.bin"
            output_model_file = os.path.join(path_to_save_model+'\\output', WEIGHTS_NAME)
            output_config_file = os.path.join(path_to_save_model+'\\output', CONFIG_NAME)
            save_dict = model.state_dict()
            torch.save(save_dict, output_model_file)
            model.config.to_json_file(output_config_file)
    else:
        print("❌ No hay adaptadores activos para fusionar.")

    tokenizer.save_pretrained(path_to_save_model+'\\output')



    # ---------------Clean---------------------------------------
    model = None;
    tokenizer = None;
    trainer = None;
    save_dict = None;

# -------------------------------------Save GGUF Format-----------------------------------------------------------
    os.mkdir(path_to_save_model+'\\model\\') 
    conversion_gguf = llama.llamaCPP_python(path_to_save_model+'\\output', path_to_save_model+'\\model\\'+path_to_save_file)
    conversion_gguf.save_model_gguf()

    # -------------------------
    # 6. Cargar modelo GGUF y hacer inferencia con llama_cpp
    # -------------------------
    import llama_cpp as Llama
    TEST_PROMPT = "Buenos días, Orbital!"
    print("🧠 Cargando modelo GGUF con llama_cpp...")
    llm = Llama(model_path=path_to_save_model+'\\model\\'+path_to_save_file, n_ctx=2048)

    print(f"💬 Prompt: {TEST_PROMPT}")
    output = llm(TEST_PROMPT, max_tokens=100, stop=["</s>"])
    print("📤 Respuesta generada:")
    print(output["choices"][0]["text"].strip())


if __name__ == "__main__":
    main()