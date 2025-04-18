import json
import os
import datetime

from pathlib import Path
import torch
import logging
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoConfig
from transformers import AutoModelForCausalLM
import transformers

class model_chat:
    def __init__(self):
            os.environ['CUDA_LAUNCH_BLOCKING']='1'
            os.environ['TORCH_USE_CUDA_DSA'] = 'True'
            self.messages=[]
            self.chat_history = []
            self.chat_history_ids = None
            self.chat_history_txt =""
            self.chat_messages = []
            
            # self.file_path_context_txt = Path("C:\\Users\\Emilio\\Documents\\GitHub\\IA\\src\\context_db\\context_txt.txt")
            self.file_path_context_json = Path("C:\\Users\\Emilio\\Documents\\GitHub\\IA\\src\\context_db\\context_v1.json")
            self.path_to_model = 'C:\\Users\\Emilio\\Documents\\GitHub\\IA\\src\\llm\\llama32_orbital_chat_3B_q4'
            # self.path_to_model = 'C:\\Users\\Emilio\\Documents\\GitHub\\IA\\src\\llm\\llama4_SCOUT_orbital_chat_17B_16E'
            
            # self.path_to_model = "meta-llama/Llama-3.2-3B-Instruct"
            
# ,torch_dtype=torch.float16, device_map='auto') 
# Model    
            torch_dtype = torch.bfloat16

            torch.cuda.empty_cache()

            # from transformers import set_seed
            # set_seed(42)
            # state_dict = torch.load(self.path_to_model, map_location='cpu')
            quantization_config = BitsAndBytesConfig(
                    # load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_quant_storage=torch_dtype,
                    # llm_int8_enable_fp32_cpu_offload=True,
                )
            torch.cuda.empty_cache()
            model_config = AutoConfig.from_pretrained(self.path_to_model)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.path_to_model,
                quantization_config=quantization_config,
                low_cpu_mem_usage=True,
                device_map='auto',
                config=model_config,
                ignore_mismatched_sizes=True
            )

            self.tokenizer = AutoTokenizer.from_pretrained(self.path_to_model, fast_tokenizer=True)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.messages = [
                {"role": "system", "content": "Tu nombre es Orbital, un asistente virtual", "timestamp": str(datetime.datetime.now().today().strftime("%d-%m-%Y %H:%M:%S"))},
                {"role": "user", "content": "Hola, Orbital", "timestamp": str(datetime.datetime.now().today().strftime("%d-%m-%Y %H:%M:%S"))},
                {"role": "assistant", "content": "Hola, mi nombre es Orbital, un asistente virtual listo para ayudarte.", "timestamp": str(datetime.datetime.now().today().strftime("%d-%m-%Y %H:%M:%S"))},
            ]

            # # set chat template
            tokenized_chat = self.tokenizer.apply_chat_template(self.messages, return_tensors="pt")
            #set device CPU
            self.device=torch.device('cpu',0)
            # self.context_db(self.file_path_context_json)
            # ----------------------------------CHAT TEMPLATE---------------------------------
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

            # self.tokenizer = AutoTokenizer.from_pretrained(self.path_to_model, token="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")
            self.tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE
            
            # ------------------------------------------------------------------------------------
            # self.model.float()
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.pipeline = transformers.pipeline(
                model=self.model,
                tokenizer=self.tokenizer,
                task='text-generation',
                torch_dtype=torch.bfloat16,     
                device_map='cpu',
            )
            #Get Context chat
            self.context_db(self.file_path_context_json)

    def append_context(self, new_file_path:Path):
         if new_file_path.is_file():
            newfile = open(new_file_path)
            content = newfile.read()
            newfile.close
            # self.chat_history.append(content)
            self.chat_history_txt = content+"\n"+self.chat_history_txt
            print("Database TXT update successfully!")
            # return self.chat_history_txt

    def append_context_json(self, new_file_path:Path):
        if new_file_path.is_file():
            # with open(new_file_path, 'r', encoding='latin1') as json_file:
            #     data = json.load(json_file)

            # # Append new data
            # # self.chat_history.append(data['content'])
            # data.append(self.chat_history)
            # self.chat_history = data

            # Write updated data back to the file
            with open(new_file_path, 'w', encoding='latin1') as file:
                file.write("")
                json.dump(self.chat_history, file, indent=4)

            print("Database update successfully!")
            # return data


    def context_db(self, path_chat_history:Path):
        # Load the JSON file
        if path_chat_history.is_file():
            with open(path_chat_history, 'r', encoding='latin1') as json_file:
                context_data = json.load(json_file)
        # Convert the JSON data to a string
        context_string = json.dumps(context_data)   
        # self.chat_history_ids = self.tokenizer.encode(context_string+ self.tokenizer.eos_token, return_tensors="pt")
        # Tokenize the context string
        inputs_id = self.tokenizer(context_string, return_tensors='pt')
        
        # self.chat_history_ids.to('cpu')
        # # Generate a response using the mode
        # # if torch.isnan(self.chat_history_ids['input_ids']).any() or torch.isinf(self.chat_history_ids['input_ids']).any():
        # #     print("Found NaN or Inf")
        # inputs_id = torch.cat([self.chat_history_ids, inputs['input_ids']], dim=-1) if self.chat_history_ids != None else inputs['input_ids']
        
        output = self.chat_history_ids = self.model.generate(
            inputs_id['input_ids'].to(self.device),
            attention_mask=inputs_id['attention_mask'].to(self.device),
            renormalize_logits=False,
            max_new_tokens=True,
        )
        print(output)
        inputs_id = None;
        # for message in context_string:
        #     # if len(message)>0:
        #     #     if len(message)>1:
        #     #         for newmess in message:
        #     #             self.chat_history.append(newmess)
        #     self.chat_history.append(message)
        # print(self.tokenizer.decode(output[0], skip_special_tokens=True))
        self.chat_history = context_string

    def freeze(self, d):
        if isinstance(d, dict):
            return frozenset((key, self.freeze(value)) for key, value in d.items())
        elif isinstance(d, list):
            return tuple(self.freeze(value) for value in d)
        return d
    
    # -----------------------------------------------------TODO -----------------------------------------------------------------------------
    def generate_reponse(self, text):
        # encode the input and add end of string token
        # input_ids = self.tokenizer.encode(text + self.tokenizer.eos_token, return_tensors="pt", padding=True)
        # Combine chat history into a single string
        self.chat_messages.append(f"user:  {text}")
        # history_str = "\n".join(self.chat_messages)
        # input_ids = self.tokenizer(text+ self.tokenizer.eos_token, return_tensors="pt", return_attention_mask=True)
        # concatenate new user input with chat history (if there is)
        # bot_input_ids = torch.cat([self.chat_history_ids, input_ids], dim=-1) if self.chat_history_ids != None else input_ids
        # generate a bot response
        
        output = self.pipeline(self.chat_history, temperature=0.7,num_beams=4, repetition_penalty=1, top_k =50, top_p=0.9)
        llm_response_text = output[0]['generated_text'][-1]['content']
        # print(output[0]['generated_text'][len(output)]['content'])
        # self.chat_history_ids = self.model.generate(
        #     input_ids['input_ids'],
        #     # input_ids=bot_input_ids['input_ids'],
        #     attention_mask=input_ids['attention_mask'],
        #     # renormalize_logits=False,
        #     # use_cache=True,
        #     # max_length = 50,
        #     # min_length = len(input_ids),
        #     # max_new_tokens=True,
        #     # return_dict_in_generate=False,
        #     do_sample=True,
        #     top_p=0.95,
        #     top_k=0,
        #     temperature=0.7,
        #     num_return_sequences=1,
        #     pad_token_id=self.tokenizer.pad_token_id
        # )
        # #print the output
        # output = self.tokenizer.decode(self.chat_history_ids[0], skip_special_tokens=True)
        
        # output = self.tokenizer.decode(self.chat_history_ids[0], skip_special_tokens=True)
        # print(f"Orbital : {output[0].split("-")[3]}")
        print(llm_response_text)
        self.chat_messages.append(f"assistant:  {llm_response_text}")
        return llm_response_text
        # return output.split("-")[3]
    # ----------------------------------------------------------------------------------------------------------------------------------------
    
    def chat_with_ollama_history(self):
        # self.context_db()
        now = datetime.datetime.now()
        var_continue = True
        while var_continue:
            user_input = input('User: ')
            
            if user_input.lower().__contains__("Orbital, apaga") or user_input.lower().__contains__("stop conversation") or user_input.lower().__contains__("finaliza conversacion")or user_input.lower().__contains__("finaliza conversación")or user_input.lower().__contains__("exit"):
                var_continue= False;
                break;


            self.chat_history.append({'role':'user', 'content': str(user_input), "timestamp": str(datetime.datetime.now().today().strftime("%d-%m-%Y %H:%M:%S"))})
            response = self.generate_reponse(user_input)

            # Add the response to the messages to maintain the history
            # self.chat_history.append(response['message'])
            # self.chat_history.append({'role':'assistant', 'content': str(datetime.datetime.now().today().strftime("%d-%m-%Y %H:%M:%S")) + " - " + response.message.content})
            # self.chat_history_txt += "User: "+user_input+"\n"+"Orbital: "+ str(datetime.datetime.now().today().strftime("%d-%m-%Y %H:%M:%S")) + " - " + response.message.content
            # print("Orbital: " + response.message.content + '\n')

            self.chat_history.append({'role':'assistant', 'content': str(response), "timestamp": str(datetime.datetime.now().today().strftime("%d-%m-%Y %H:%M:%S"))})
            print("Orbital: " + response + '\n')

        if user_input.lower().__contains__("base de datos") or user_input.lower().__contains__("database"):
                if user_input.lower().__contains__("almacena") or user_input.lower().__contains__("insert") or user_input.lower().__contains__("guarda"):
                    # newFilePath = self.file_path_context_txt
                    newFilePath_json = self.file_path_context_json
                    # if os.path.exists(newFilePath):
                    #     context_txt = self.append_context(newFilePath)
                    if os.path.exists(newFilePath_json):
                        self.append_context_json(newFilePath_json)
                    else:
                        with open(newFilePath_json, 'w+', encoding='latin1') as file:
                            json.dump(self.chat_history, file, indent=4)
                    

                    # newfile  = open(newFilePath, "w+")
                    # newfile.write(self.chat_history_txt)
                    # newfile.close()
            