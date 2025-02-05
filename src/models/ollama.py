import requests
import json
import ollama

from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
#import chainlit as cl

#get memory
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import conversation
import os
import datetime

from pathlib import Path
    

class OllamaChat:
    def __init__(self, qst):
            self.url = "http://localhost:11434/api/generate"
            self.headers = {
                "Content-Type" : "application/json"
            }
            self.data = {
                "model": "mistral",
                "prompt": qst,
                "stream":True
            }
            self.resultQuery = "{}"
            self.model = "llama3.2"
            self.content = [{'role':'user', 'content':qst}]
            self.memory = ConversationBufferMemory()
            self.messages=""
            self.chat_history = []
    
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
            
    def getReponseOllamaChat(self,query:str):
        new_message = [{'role':'user', 'content':query}]
        new_message.append(self.content)
        reponse = ollama.chat(
            model = self.model,
            messages = self.content,
            options={
                    'num_ctx': 4096
                }
        )
        self.content = new_message.append({'role':'system', 'content':reponse})
        
        self.resultQuery= reponse.message.content
        print(self.resultQuery)
    
    def append_context(self, new_file_path:Path):   
         if new_file_path.is_file():
            newfile = open(new_file_path)
            content = newfile.read()
            newfile.close
            self.chat_history.append(content)
            
    def chat_with_ollama_history(self, user_input):
        #create context IA"
        folder_path = Path("C:\\Users\\Emilio Guillem\\Documents\\GIT\\IA\\src\\context_db")
        for file_path in folder_path.iterdir():
            self.append_context(file_path)
                
                
        while True:
            user_input = input('Emilio: ')

            if user_input.lower().__contains__("Orbital, apaga") or user_input.lower().__contains__("stop conversation") or user_input.lower().__contains__("finaliza conversacion")or user_input.lower().__contains__("finaliza conversación")or user_input.lower().__contains__("cerrar"):
                break
            if user_input.lower().__contains__("base de datos") or user_input.lower().__contains__("database"):
                if user_input.lower().__contains__("almacena") or user_input.lower().__contains__("insert") or user_input.lower().__contains__("guarda"):
                    now = datetime.datetime.now()
                    newFilePath = "C:\\Users\\Emilio Guillem\\Documents\\GIT\\IA\\src\\context_db\\context_"+str(now.strftime("%d%m%Y"))
                    if os.path.exists(newFilePath):
                        self.append_context(newFilePath)
                        
                    newfile  = open(newFilePath, "w+")
                    newfile.write(''.join(str(x) for x in self.chat_history))
                    newfile.close()
                       
            self.chat_history.append({'role':'user', 'content':user_input})
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
            self.chat_history.append(response['message'])
            # self.chat_history.append({'role':'assistant', 'content':response['message']})
            print("Orbital: " + response.message.content + '\n')