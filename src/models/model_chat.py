import json
import os
import datetime

from pathlib import Path
import torch
import logging
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import transformers

class model_chat:
    def __init__(self):
            os.environ['CUDA_LAUNCH_BLOCKING']='1'
            os.environ['TORCH_USE_CUDA_DSA'] = 'True'
            self.messages=""
            self.chat_history = []
            self.chat_history_ids = None
            self.chat_history_txt =""
            self.file_path_context_txt = Path("C:\\Users\\Emilio\\Documents\\GitHub\\IA\\src\\context_db\\context_txt.txt")
            self.file_path_context_json = Path("C:\\Users\\Emilio\\Documents\\GitHub\\IA\\src\\context_db\\context.json")
            
            torch.inference_mode()
            torch.cuda.empty_cache()
            

            self.tokenizer = AutoTokenizer.from_pretrained('C:\\Users\\Emilio\\Documents\\GitHub\\IA\\TrainingTest_3B')
# ,torch_dtype=torch.float16, device_map='auto') 
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained('C:\\Users\\Emilio\\Documents\\GitHub\\IA\\TrainingTest_3B',
                                                              torch_dtype=torch.bfloat16, device_map='auto') 
            # from transformers import set_seed
            # set_seed(42)

            self.model.float()
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.messages = [
                {"role": "system", "content": "Tu nombre es Orbital, un asistente virtual"},
                {"role": "user", "content": "Buenos días, Orbital"},
                {"role": "assistant", "content": "Buenos días, mi nombre es Orbital"}
            ]

            # set chat template
            tokenized_chat = self.tokenizer.apply_chat_template(self.messages, return_tensors="pt")
            #set device GPU
            self.device=torch.device('cuda',0)
            self.context_db(self.file_path_context_json)
            # self.pipeline = transformers.pipeline(
            #     model=self.model,
            #     tokenizer=self.tokenizer,
            #     task='text-generation',
            #     torch_dtype=torch.bfloat16,     
            #     device_map='auto',
            # )

    def append_context(self, new_file_path:Path):
         if new_file_path.is_file():
            newfile = open(new_file_path)
            content = newfile.read()
            newfile.close
            # self.chat_history.append(content)
            self.chat_history_txt = content+"\n"+self.chat_history_txt
            print("Database TXT update successfully!")
            return self.chat_history_txt

    def append_context_json(self, new_file_path:Path):
        if new_file_path.is_file():
            with open(new_file_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)

            # Append new data
            # self.chat_history.append(data['content'])
            data.append(self.chat_history)
            self.chat_history = data

            # Write updated data back to the file
            with open(new_file_path, 'w+') as file:
                json.dump(self.chat_history, file, indent=4)

            print("Database update successfully!")
            return self.chat_history


    def context_db(self, path_chat_history:Path):
        # Load the JSON file
        if path_chat_history.is_file():
            with open(path_chat_history, 'r', encoding='utf-8') as json_file:
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
            se_cache=True,
            max_length=100,
        )
        inputs_id = None;
        # print(self.tokenizer.decode(output[0], skip_special_tokens=True))



    def generate_reponse(self, text):
        # encode the input and add end of string token
        # device=torch.device('cuda',0)
        # input_ids = self.tokenizer.encode(text + self.tokenizer.eos_token, return_tensors="pt", padding=True)
        input_ids = self.tokenizer(text+ self.tokenizer.eos_token, return_tensors="pt", padding=True)
        # concatenate new user input with chat history (if there is)
        bot_input_ids = torch.cat([self.chat_history_ids, input_ids], dim=-1) if self.chat_history_ids != None else input_ids
        bot_input_ids = bot_input_ids.to(self.device)
        # generate a bot response
        # newOutput = self.pipeline.predict(text)
        # print(newOutput)
        self.chat_history_ids = self.model.generate(
            input_ids=bot_input_ids['input_ids'].to(self.device),
            attention_mask=bot_input_ids['attention_mask'].to(self.device),
            # input_ids = bot_input_ids.to(device),
            renormalize_logits=False,
            use_cache=True,
            # max_length=4096,
            max_new_tokens=True,
            do_sample=True,
            top_p=0.95,
            top_k=0,
            temperature=0.7,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.pad_token_id
        )
        # #print the output
        # output = self.tokenizer.decode(self.chat_history_ids[0], skip_special_tokens=True)
        # generator = self.pipeline
        # output = generator(text, renormalize_logits=True, do_sample=True, use_cache=True, max_new_tokens=True, temperature=0.7)
        output = self.tokenizer.batch_decode(self.chat_history_ids, skip_special_tokens=True)
        # print(f"Orbital : {output[0].split("-")[3]}")
        return output[0].split("-")[3]
        
    
    def chat_with_ollama_history(self):
        # self.context_db()
        now = datetime.datetime.now()
        var_continue = True
        while var_continue:
            user_input = input('User: ')
            user_input = str(datetime.datetime.now().today().strftime("%d-%m-%Y %H:%M:%S")) + " - " + user_input
            if user_input.lower().__contains__("Orbital, apaga") or user_input.lower().__contains__("stop conversation") or user_input.lower().__contains__("finaliza conversacion")or user_input.lower().__contains__("finaliza conversación")or user_input.lower().__contains__("shutdown"):
                var_continue= False;
                break;
            if user_input.lower().__contains__("base de datos") or user_input.lower().__contains__("database"):
                if user_input.lower().__contains__("almacena") or user_input.lower().__contains__("insert") or user_input.lower().__contains__("guarda"):
                    newFilePath = self.file_path_context_txt
                    newFilePath_json = self.file_path_context_json
                    if os.path.exists(newFilePath):
                        context_txt = self.append_context(newFilePath)
                    if os.path.exists(newFilePath_json):
                        context_json = self.append_context_json(newFilePath_json)
                    else:
                        with open(newFilePath_json, 'w+') as file:
                            json.dump(self.chat_history, file, indent=4)
                    

                    newfile  = open(newFilePath, "w+")
                    newfile.write(self.chat_history_txt)
                    newfile.close()

            self.chat_history.append({'role':'user', 'content':str(datetime.datetime.now().today().strftime("%d-%m-%Y %H:%M:%S")) + " - " + user_input})
            response = self.generate_reponse(user_input)

            # Add the response to the messages to maintain the history
            # self.chat_history.append(response['message'])
            # self.chat_history.append({'role':'assistant', 'content': str(datetime.datetime.now().today().strftime("%d-%m-%Y %H:%M:%S")) + " - " + response.message.content})
            # self.chat_history_txt += "User: "+user_input+"\n"+"Orbital: "+ str(datetime.datetime.now().today().strftime("%d-%m-%Y %H:%M:%S")) + " - " + response.message.content
            # print("Orbital: " + response.message.content + '\n')






            


            self.chat_history.append({'role':'assistant', 'content': str(datetime.datetime.now().today().strftime("%d-%m-%Y %H:%M:%S")) + " - " + 
                                      response})
            print("Orbital: " + response + '\n')


            