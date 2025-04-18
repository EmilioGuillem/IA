# import openai

# openai.api_key = ""

# modelos = openai.Model.list()

# modelo = "text-davinci-002"
# prompt = "¿Cuál es la capital de Francia?"

# respuesta = openai.Completion.create(engine=modelo, prompt=prompt, n=1)

# text = respuesta.choices[0].text.strip()B
# print(text)
from pathlib import Path
import model_chat
import Ollama_chat
import LMStudio_chat


def main():

    # newOllamaConnexion = Ollama_chat.OllamaChat()

    # newOllamaConnexion.chat_with_ollama_history()

    newConnexion = LMStudio_chat.LMChat()
    newConnexion.chat_with_lmstudio_history()

if __name__ == "__main__":
    main()