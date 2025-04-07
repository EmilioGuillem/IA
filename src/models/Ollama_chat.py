import requests
import json
import ollama

# from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain_community.adapters.openai import convert_dict_to_message
#import chainlit as cl

from langchain.chains import conversation
import os
import datetime
from collections import deque

from pathlib import Path

class OllamaChat:
    def __init__(self):
            self.url = "http://localhost:11434/api/generate"
            self.headers = {
                "Content-Type" : "application/json"
            }
            self.data = {
                "model": "mistral",
                "prompt": "",
                "stream":True
            }
            self.resultQuery = "{}"
            # self.model = "llama3.2"
            self.model = "orbital"
            self.messages=""
            self.chat_history = []
            self.chat_history_txt =""
            self.context_db()

    #Suprimir
    # def getReponse(self, data:json):
    #     response = requests.post(url=self.url, headers=self.headers, data=json.dumps(self.data))
    #     if response.status_code==200:
    #         response_text = response.text
    #         data = json.loads(response_text)
    #         actual_response = data["response"]
    #         print(actual_response)
    #         self.resultQuery = actual_response
    #     else:
    #         print("Error: ", response.status_code, response.text)

    def append_context(self, new_file_path:Path):
         if new_file_path.is_file():
            newfile = open(new_file_path)
            content = newfile.read()
            newfile.close
            # self.chat_history.append(content)
            self.chat_history_txt = content+"\n"+self.chat_history_txt
            print("Database TXT update successfully!")

    def append_context_json(self, new_file_path:Path):
        if new_file_path.is_file():
            # with open(new_file_path, 'r', encoding='latin1') as json_file:
            #     data = json.load(json_file)

            # # Append new data
            # # self.chat_history.append(data['content'])
            # # data.append(self.chat_history)
            # # data = deque(data)
            # data.append(self.chat_history)

            # Write updated data back to the file
            with open(new_file_path, 'w', encoding='latin1') as file:
                file.write("")
                json.dump(self.chat_history, file, indent=4)

            print("Database update successfully!")
            # return data


    def context_db(self):
        #create context IA"
        # message_history_json = []
        # file_path = Path("C:\\Users\\Emilio\\Documents\\GitHub\\IA\\src\\context_db\\context_txt.txt")
        file_path_json = Path("C:\\Users\\Emilio\\Documents\\GitHub\\IA\\src\\context_db\\context_v1.json")
        # for file_path in folder_path.iterdir():
        with open(file_path_json, 'r', encoding='latin1') as json_file:
                data = json.load(json_file)
        
        # self.chat_history.append({'role':'user', 'content':'Aquí tienes el histórico de conversaciones anteriores', "timestamp": str(datetime.datetime.now().today().strftime("%d-%m-%Y %H:%M:%S"))})
        
        for message in data:
            # if len(message)>0:
            #     if len(message)>1:
            #         for newmess in message:
            #             self.chat_history.append(newmess)
            self.chat_history.append(message)
        
        # response = ollama.chat(
        #     model = self.model,
        #     messages=message_history,
        #     options={
        #             'num_ctx': 4096,
        #             'temperature': 0.7,
        #             'repeat_penalty':1.2
        #         },
        # )
        reponse = ollama.chat(
            model = self.model,
            messages=self.chat_history,
            options={
                    'num_ctx': 4096,
                    'temperature': 0.7,
                    'repeat_penalty':1.2
                },
        )
    def chat_with_ollama_history(self):
        # self.context_db()
        now = datetime.datetime.now()
        var_continue = True
        while var_continue:
            user_input = input('User: ')
            if user_input.lower().__contains__("Orbital, apaga") or user_input.lower().__contains__("stop conversation") or user_input.lower().__contains__("finaliza conversacion")or user_input.lower().__contains__("finaliza connversación")or user_input.lower().__contains__("exit"):
                var_continue= False;
                break;

            self.chat_history.append({'role':'user', 'content':str(user_input), "timestamp": str(datetime.datetime.now().today().strftime("%d-%m-%Y %H:%M:%S"))})
            response = ollama.chat(
                model = self.model,
                messages=self.chat_history,
                options={
                        'num_ctx': 4096,
                        'temperature': 0.7,
                        'repeat_penalty':1.2
                    },
            )

            # Add the response to the messages to maintain the history
            # self.chat_history.append(response['message'])
            self.chat_history.append({'role':'assistant', 'content': str(response.message.content), "timestamp": str(datetime.datetime.now().today().strftime("%d-%m-%Y %H:%M:%S"))})
            # self.chat_history_txt += "User: "+user_input+"\n"+"Orbital: "+ response.message.content+"\n"
            print("Orbital: " + response.message.content + '\n')

            if user_input.lower().__contains__("base de datos") or user_input.lower().__contains__("database"):
                if user_input.lower().__contains__("almacena") or user_input.lower().__contains__("insert") or user_input.lower().__contains__("guarda"):
                    # newFilePath = Path("C:\\Users\\Emilio\\Documents\\GitHub\\IA\\src\\context_db\\context_txt.txt")
                    newFilePath_json = Path("C:\\Users\\Emilio\\Documents\\GitHub\\IA\\src\\context_db\\context_v1.json")
                    # if os.path.exists(newFilePath):
                    #     self.append_context(newFilePath)
                    if os.path.exists(newFilePath_json):
                        self.append_context_json(newFilePath_json)
                    else:
                        with open(newFilePath_json, 'w', encoding='latin1') as file:
                            json.dump(self.chat_history, file, indent=4)
                    

                    # newfile  = open(newFilePath, "w+")
                    # newfile.write(self.chat_history_txt)
                    # newfile.close()