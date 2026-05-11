# import openai

# openai.api_key = ""

# modelos = openai.Model.list()

# modelo = "text-davinci-002"
# prompt = "¿Cuál es la capital de Francia?"

# respuesta = openai.Completion.create(engine=modelo, prompt=prompt, n=1)

# text = respuesta.choices[0].text.strip()B
# print(text)
from pathlib import Path
# import model_chat
# import Ollama_chat
# import LMStudio_chat


def main():

    # newOllamaConnexion = Ollama_chat.OllamaChat()

    # newOllamaConnexion.chat_with_ollama_history()

    # newConnexion = LMStudio_chat.LMChat()
    # newConnexion.chat_with_lmstudio_history()
    # -------------------------
    # 6. Cargar modelo GGUF y hacer inferencia con llama_cpp
    # -------------------------
    from llama_cpp import Llama as llmcpp
    path_to_save_model = 'C:\\Users\\Emilio\\Documents\\GitHub\\IA\\src\\llm\\test\\'
    path_to_save_file = r'Llama-orbital-3.2-3B-Instruct-Q4_K_M.gguf'
    TEST_PROMPT = "Buenos días, Orbital!"
    print("🧠 Cargando modelo GGUF con llama_cpp...")
    llm = llmcpp(model_path=path_to_save_model+'model\\'+path_to_save_file, n_ctx=2048)

    print(f"💬 Prompt: {TEST_PROMPT}")
    output = llm(TEST_PROMPT, max_tokens=100, stop=["</s>"])
    print("📤 Respuesta generada:")
    print(output["choices"][0]["text"].strip())

if __name__ == "__main__":
    main()