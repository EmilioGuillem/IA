# Claude Code Python Model Scaffold

Este directorio contiene un scaffold para usar `claude-code` como referencia y construir un modelo Python compatible con Hugging Face.

## Qué incluye

- `requirements.txt`: dependencias Python para entrenar y ejecutar el modelo.
- `prompt_template.py`: prompt base extraído de la lógica de `claude-code` y formato de conversación recomendado.
- `train.py`: script de fine-tuning para un modelo de lenguaje causal compatible con Hugging Face.
- `chat.py`: cliente de chat interactivo para usar el modelo entrenado.
- `data/sample_training_data.jsonl`: ejemplo de datos para entrenamiento en formato JSONL.

## Cómo usar

1. Instala dependencias:

```bash
python -m pip install -r python-model/requirements.txt
```

2. Prepara datos de entrenamiento en `python-model/data/*.jsonl`.

3. Fine-tunea un modelo base:

```bash
python python-model/train.py \
  --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
  --dataset_path python-model/data/sample_training_data.jsonl \
  --output_dir python-model/fine-tuned-model
```

4. Inicia el chat con el modelo guardado:

```bash
python python-model/chat.py --model_path python-model/fine-tuned-model
```

5. Si quieres cargar un modelo local ya disponible y usar los prompts extraídos de Claude Code, usa el wrapper local:

```bash
python python-model/claude-model/local_claude_model.py --model_path /ruta/al/modelo/local
```

## Nota importante

Este directorio no contiene pesos propietarios de Claude ni un modelo de Anthropic. Proporciona una estructura de entrenamiento y chat basada en las directrices y prompts extraídos del repositorio `claude-code`.

## Nuevos scripts añadidos

- `fetch_model.py`: descarga o realiza snapshot de un repo de Hugging Face a un directorio local.
- `finetune.py`: fine-tunea un modelo causal usando un archivo JSONL local; soporta LoRA si `peft` está instalado. Si no se pasa `--train_file`, `finetune.py` combinará automáticamente los archivos diarios de conversaciones (ver más abajo).
- `offline_chat.py`: carga un modelo local (base o fine-tuned) y abre un bucle de chat offline.

Ejemplos rápidos:

```bash
# descargar repo HF (opcional)
python python-model/fetch_model.py --repo_id DavidAU/Qwen3.6-40B-Claude-4.6-Opus-Deckard-Heretic-Uncensored-Thinking-NEO-CODE-Di-IMatrix-MAX-GGUF --local_dir ./models/qwen

# fine-tune con un JSONL local (o sin --train_file para usar todos los archivos diarios de conversaciones)
python python-model/finetune.py --model_path ./models/qwen --train_file ./python-model/data/sample_training_data.jsonl --output_dir ./models/qwen-finetuned --use_lora

# chat offline
python python-model/offline_chat.py --model_dir ./models/qwen-finetuned --device cpu
```

Archivos de conversaciones diarios

- Los intercambios se almacenan en `python-model/data` como archivos diarios: `conversations-YYYY-MM-DD.jsonl`.
- Para entrenar usando todo el histórico, `finetune.py` combina esos archivos diarios en `python-model/data/combined_training.jsonl` si no se proporciona `--train_file`.

Opciones útiles:

- Para normalizar los diarios en un archivo de entrenamiento `prompt/response` antes del fine-tuning use `--normalize`:

```bash
python python-model/finetune.py --model_path ./models/qwen --output_dir ./models/qwen-finetuned --normalize
```

- Para archivar archivos de conversaciones más antiguos que N días após la combinación use `--archive_days N`:

```bash
python python-model/finetune.py --model_path ./models/qwen --output_dir ./models/qwen-finetuned --archive_days 30
```

Notas:

- Los modelos grandes requieren mucha RAM/GPU; para hardware limitado usa LoRA y bitsandbytes (entrenamiento en 8-bit).
- Para usar LoRA instala `peft` y configura `accelerate` si es necesario.
- Este proyecto no incluye pesos propietarios de Claude; debes descargar o proporcionar un modelo HF compatible en local.
