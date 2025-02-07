# import openai

# openai.api_key = ""

# modelos = openai.Model.list()

# modelo = "text-davinci-002"
# prompt = "¿Cuál es la capital de Francia?"

# respuesta = openai.Completion.create(engine=modelo, prompt=prompt, n=1)

# text = respuesta.choices[0].text.strip()
# print(text)
from pathlib import Path
import Ollama


def main():
    
    
    q = "Hola!"
    newOllamaConnexion = Ollama.OllamaChat(q)
    # newOllamaConnexion.getReponseOllamaChat(q)
    # # newOllamaConnexion.chat_with_ollama(q)
    # while True:
    #     input_user = input("")
    #     newOllamaConnexion = Ollama.OllamaChat(input_user)
    #     newOllamaConnexion.getReponseOllamaChat(input_user)
    # newOllamaConnexion.chat_with_ollama_history(q)
    newOllamaConnexion.chat_history.append({
            "role": "user",
            "content": "07-02-2025 13:22:51: Esto es un test de formato Json"})
    newOllamaConnexion.append_context_json(Path("src\context_db\context.json"))
if __name__ == "__main__":
    main()