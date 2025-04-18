import requests
import json
import lmstudio as lms

import os
import datetime

from pathlib import Path

class LMChat:
    def __init__(self):
            self.loaded_models = lms.list_loaded_models()

            for idx, model in enumerate(self.loaded_models):
               print(f"{idx:>3} {model}")
            
            self.model = self.loaded_models[0]

            # self.model = "llama3.2"
            # self.model = "orbital"
            self.messages = [
                {"role": "system", "content": "Tu nombre es Orbital, un asistente virtual", "timestamp": str(datetime.datetime.now().today().strftime("%d-%m-%Y %H:%M:%S"))},
                {"role": "user", "content": "Hola, Orbital", "timestamp": str(datetime.datetime.now().today().strftime("%d-%m-%Y %H:%M:%S"))},
                {"role": "assistant", "content": "Hola, mi nombre es Orbital, un asistente virtual listo para ayudarte.", "timestamp": str(datetime.datetime.now().today().strftime("%d-%m-%Y %H:%M:%S"))},
            ]

            self.chat = lms.Chat(initial_prompt="Tu nombre es Orbital, un asistente virtual")
            self.chat_history = []
            self.context_db()


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
            with open(new_file_path, 'w', encoding='latin1') as file:
                file.write("")
                json.dump(self.chat_history, file, indent=4)

            print("Database update successfully!")
            # return data


    def context_db(self):
        #create context IA"
        file_path_json = Path("C:\\Users\\Emilio\\Documents\\GitHub\\IA\\src\\context_db\\context_v1.json")
        # for file_path in folder_path.iterdir():
        with open(file_path_json, 'r', encoding='latin1') as json_file:
                data = json.load(json_file)

        for message in data:
            self.chat_history.append(message)

        self.chat = self.chat.from_history({"messages": self.chat_history})

    def chat_with_lmstudio_history(self):
        # self.context_db()
        now = datetime.datetime.now()
        var_continue = True
        while var_continue:
            user_input = input('User: ')
            if user_input.lower().__contains__("Orbital, apaga") or user_input.lower().__contains__("stop conversation") or user_input.lower().__contains__("finaliza conversacion")or user_input.lower().__contains__("finaliza connversación")or user_input.lower().__contains__("exit"):
                var_continue= False;
                break;

            self.chat_history.append({'role':'user', 'content':str(user_input), "timestamp": str(datetime.datetime.now().today().strftime("%d-%m-%Y %H:%M:%S"))})
            self.chat.add_user_message(str(user_input))
            response = self.model.respond(self.chat)
            self.chat.add_assistant_response(response.content)
            # Add the .content to the messages to maintain the history
            # self.chat_history.append(response['message'])
            self.chat_history.append({'role':'assistant', 'content': str(response.content), "timestamp": str(datetime.datetime.now().today().strftime("%d-%m-%Y %H:%M:%S"))})
            # self.chat_history_txt += "User: "+user_input+"\n"+"Orbital: "+ response.message.content+"\n"
            print("Orbital: " + response.content + '\n')

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