import os
import subprocess


class llamaCPP_python:
    # -------------------------
    # RUTAS - AJÚSTALAS A TU CASO
    # -------------------------
    def __init__(self, model_dir, model_gguf):
        
        self.LLAMA_CPP_DIR = "C:\\Users\\Emilio\\llama.cpp"
        self.MODEL_DIR = model_dir  # ruta al modelo fusionado con LoRA (con config.json y pytorch_model.bin)
        # self.VOCAB_DIR = model_base      # ruta al modelo original descargado de Hugging Face (con tokenizer.model, etc.)
        self.OUTPUT_GGUF = model_gguf

    def save_model_gguf(self):
        # -------------------------
        # Paso 2: Clonar llama.cpp si no existe
        # -------------------------
        if not os.path.exists(self.LLAMA_CPP_DIR):
            subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp"], check=True)
            print("✅ Repositorio llama.cpp clonado.")

        # -------------------------
        # Paso 3: Ejecutar convert.py
        # -------------------------
        convert_script = os.path.join(self.LLAMA_CPP_DIR, "convert_hf_to_gguf.py")

        cmd = [
            "C:\\Program Files\\Python312\\python.exe", convert_script,
            self.MODEL_DIR,
            "--outfile", self.OUTPUT_GGUF
        ]

        print("🚀 Ejecutando conversión a GGUF...")
        subprocess.run(cmd, check=True)
        print(f"✅ Conversión completada: {self.OUTPUT_GGUF}")
