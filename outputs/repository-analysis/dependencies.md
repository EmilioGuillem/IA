---
workflow: brownfield-analysis
trigger: user
date: '2026-09-22'
status: draft
inputDocuments:
  - sopra_clockin/requirements.txt
  - sopra_clockin/src/sopra_clockin.py
  - sopra_clockin/config/config.py
  - claude-code/python-model/requirements.txt
  - claude-code/python-model/README.md
  - Task.txt
  - .gitignore
changeHistory:
  - date: '2026-09-22'
    description: 'Mapa inicial de dependencias internas y externas'
    changes:
      - 'Relacionados componentes Python con sus bibliotecas declaradas o previstas'
      - 'Registradas dependencias externas y ausencias de manifiestos'
holisticQualityRating: not-rated
overallStatus: not-reviewed
trace_id: 4d9a8c2e-7b61-4a30-9c4f-1e6b5d8a2037
station: repository-analysis
agent: analysis-agent
skill: repo-analysis
timestamp: '2026-09-22T00:00:00Z'
---
<!-- AI_DISCLAIMER v1.0 -->
# Mapa de dependencias

> Este archivo ha sido creado total o parcialmente con la asistencia de herramientas de inteligencia artificial.
> Todo el contenido ha sido generado bajo la supervisión directa de una persona identificada,
> y bajo el marco de cumplimiento AI.Backbone Orchestrator.

## Dependencias externas

| Componente | Dependencias | Uso |
|---|---|---|
| `sopra_clockin` | `selenium>=4.0.0` | Automatización de navegador y portal web |
| `sopra_clockin` | `webdriver-manager>=3.8.0` | Gestión opcional de drivers; el código principal crea drivers Selenium directamente |
| `claude-code/python-model` | PyTorch, Transformers, Datasets | Entrenamiento e inferencia de modelos causales |
| `claude-code/python-model` | Accelerate, PEFT, bitsandbytes | Aceleración y fine-tuning, incluido LoRA |
| `claude-code/python-model` | Hugging Face Hub, safetensors | Descarga y almacenamiento de modelos |
| Prototipo `src/` | Ollama, ChromaDB, audio y reconocimiento | Dependencias previstas en `Task.txt`; no existe manifiesto verificable |
| Formación | Bibliotecas ML no verificadas | Los scripts no se pudieron clasificar por manifiesto raíz |

## Dependencias internas

```text
sopra_clockin/src/sopra_clockin.py
  -> sopra_clockin/config/config.py
  -> sopra_clockin/src/logger_config.py
  -> Selenium WebDriver
  -> portal SopraGP4U

claude-code/python-model/{chat,offline_chat,train,finetune}.py
  -> claude-code/python-model/data/*.jsonl
  -> modelo local o repositorio Hugging Face
  -> PyTorch / Transformers / Datasets

src/main.py (vacío)
  -> src/llm/*, src/models/*, src/context_db/* (relación prevista, no confirmada)
  -> Ollama / audio / ChromaDB (según Task.txt)
```

## Integraciones y recursos

- **Portal externo:** URL configurada en `sopra_clockin/config/config.py`; el flujo depende de la estructura HTML, disponibilidad de red y navegador.
- **Credenciales:** `SOPRA_USERNAME` y `SOPRA_PASSWORD` se leen desde variables de entorno, aunque `config.py` también admite valores de configuración JSON; esto merece revisión para evitar secretos persistidos.
- **Navegadores:** Chrome por defecto y Edge como alternativa.
- **Modelos:** el scaffold depende de modelos locales o de Hugging Face; no incluye pesos de modelo.
- **Persistencia experimental:** el diseño menciona ChromaDB y los módulos de `src/context_db/` incluyen adaptadores de base de datos, pero no se confirmó un flujo integrado.

## Observaciones de mantenibilidad

1. Falta un manifiesto raíz para coordinar entornos y comandos.
2. Las dependencias de subproyectos deberían instalarse en entornos virtuales separados.
3. El código TypeScript no tiene en el workspace inspeccionado un `package.json` visible, por lo que no se puede validar su build o grafo de dependencias.
4. Las dependencias previstas en `Task.txt` deben formalizarse antes de tratar el prototipo de audio como aplicación ejecutable.