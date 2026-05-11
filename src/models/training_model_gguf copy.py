# from numba import jit, cuda
import json
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
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

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


    # dataset_train, dataset_eval = load_dataset("json", data_files="C:\\Users\\Emilio\\Documents\\GitHub\\IA\\src\\context_db\\context_v1.json",encoding='latin1',  split=['train[:80%]', 'train[80%:]'])

    torch.cuda.empty_cache()
    # Model    
    torch_dtype = torch.bfloat16
    quant_storage_dtype = torch.bfloat16
    batch_size  =4
    lr = 1e-5
    epochs=3
    model_config = AutoConfig.from_pretrained(path_to_config_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(path_to_model, gguf_file=gguf_file_model, low_cpu_mem_usage=True,
        device_map='auto',torch_dtype=quant_storage_dtype, config=model_config)
    tokenizer = AutoTokenizer.from_pretrained(path_to_config_model)
    tokenizer.pad_token = tokenizer.eos_token
    # Definir el conjunto de datos
    class LanguageDataset(Dataset):
        def __init__(self, data, tokenizer):
            self.data = data
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            text = self.data[idx]
            inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
            labels = inputs["input_ids"].clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            return {
                "input_ids": inputs["input_ids"].flatten(),
                "attention_mask": inputs["attention_mask"].flatten(),
                "labels": labels.flatten()
            }
    
    #load dataset
    with open ('C:\\Users\\Emilio\\Documents\\GitHub\\IA\\src\\context_db\\context_v1.json') as f:
        json_data = json.load(f)
    data = []
    for item in json_data:
        for key, value in item.items():
            if isinstance(value, str):
                data.append(value)

    dataset = LanguageDataset(data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # Definir el optimizador y la función de pérdida
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = CrossEntropyLoss()

    ##########################
    # Train model
    ##########################

    # Entrenar el modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")

    model.eval()

    # ------------------------------------------SAVE MODEL-----------------------------------------------------------------
    model.save_pretrained(path_to_save_model+'output')
    tokenizer.save_pretrained(path_to_save_model+'output')
    output_config_file = os.path.join(path_to_save_model+'output', "config.json")
    model.config.to_json_file(output_config_file)

    torch.save(model.state_dict(),path_to_save_model+'output\\orbital_test.pth')


    # ---------------Clean---------------------------------------
    model = None;
    tokenizer = None;
    trainer = None;
    save_dict = None;

# -------------------------------------Save GGUF Format-----------------------------------------------------------
    os.mkdir(path_to_save_model+'\\model\\') 
    conversion_gguf = llama.llamaCPP_python(path_to_save_model+'output', path_to_save_model+'model\\'+path_to_save_file)
    conversion_gguf.save_model_gguf()

    # -------------------------
    # 6. Cargar modelo GGUF y hacer inferencia con llama_cpp
    # -------------------------
    from llama_cpp import Llama as llmcpp
    path_to_save_model = 'C:\\Users\\Emilio\\Documents\\GitHub\\IA\\src\\llm\\test\\'
    path_to_save_file = 'Llama-orbital-3.2-3B-Instruct-Q4_K_M.gguf'
    TEST_PROMPT = "Buenos días, Orbital!"
    print("🧠 Cargando modelo GGUF con llama_cpp...")
    llm = llmcpp(model_path=path_to_save_model+'model\\'+path_to_save_file, n_ctx=2048)

    print(f"💬 Prompt: {TEST_PROMPT}")
    output = llm(TEST_PROMPT, max_tokens=100, stop=["</s>"])
    print("📤 Respuesta generada:")
    print(output["choices"][0]["text"].strip())


if __name__ == "__main__":
    main()