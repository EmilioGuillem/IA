# Claude Model Local Wrapper

Este directorio ofrece un wrapper local para ejecutar un modelo de lenguaje en tu máquina usando los prompts y la guía de comportamiento extraídos de `claude-code`.

## Qué incluye

- `local_claude_model.py`: script que carga un modelo de Hugging Face local y usa un prompt "Claude Code style".
- `README.md`: esta documentación.

## Nota importante

No hay pesos originales de Claude en este repositorio. El repositorio contiene solo código fuente y prompts, no un modelo propietario. Este wrapper está diseñado para funcionar con un modelo local compatible de Hugging Face que tú proveas.

## Uso

1. Coloca un modelo local compatible en un directorio con `config.json`, `pytorch_model.bin`, `tokenizer.json`, etc.
2. Ejecuta:

```bash
python python-model/claude-model/local_claude_model.py --model_path /ruta/al/modelo/local
```

3. Escribe mensajes y usa el chat.
