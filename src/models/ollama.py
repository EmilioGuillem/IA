import requests
import json
import ollama

from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
#import chainlit as cl

#get memory
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import conversation
    

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
        
    
    # Use memory to store past prompts and responses
    def chat_with_ollama(self,user_input):
        # Retrieve past conversations
        newmemory = ConversationBufferMemory()
        newconver = conversation(
            llm = ollama.chat(
                model = self.model,
                messages = self.content,
                options={
                        'num_ctx': 4096
                    }
            ),
            verbose = True,
            memory=newmemory
        )     
        response = newconver(input=user_input)
        print(response)
        
        return response
    
    def chat_with_ollama_history(self, user_input):
        messages=""
        while True:
            user_input = input('Chat with history: ')
            new_message = [{'role':'user', 'content':user_input}]
            response = ollama.chat(
                model = self.model,
                messages=new_message
                + [
                {'role': 'user', 'content': user_input},
                ],
                options={
                        'num_ctx': 4096
                    },
            )

            # Add the response to the messages to maintain the history
            messages += [
                {'role': 'user', 'content': user_input},
                {'role': 'assistant', 'content': response.message.content},
            ]
            print(response.message.content + '\n')