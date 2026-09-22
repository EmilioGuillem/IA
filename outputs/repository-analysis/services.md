---
workflow: brownfield-analysis
trigger: user
date: '2026-09-22'
status: draft
inputDocuments:
  - src/main.py
  - Task.txt
  - sopra_clockin/README.md
  - sopra_clockin/src/sopra_clockin.py
  - sopra_clockin/src/test_sopra_clockin.py
  - claude-code/python-model/README.md
  - claude-code/python-model/requirements.txt
changeHistory:
  - date: '2026-09-22'
    description: 'Inventario funcional de componentes y puntos de entrada'
    changes:
      - 'Clasificados los componentes por responsabilidad y madurez observable'
      - 'Identificados puntos de entrada y pruebas disponibles'
holisticQualityRating: not-rated
overallStatus: not-reviewed
trace_id: 4d9a8c2e-7b61-4a30-9c4f-1e6b5d8a2037
station: repository-analysis
agent: analysis-agent
skill: repo-analysis
timestamp: '2026-09-22T00:00:00Z'
---
<!-- AI_DISCLAIMER v1.0 -->
# Inventario de componentes

> Este archivo ha sido creado total o parcialmente con la asistencia de herramientas de inteligencia artificial.
> Todo el contenido ha sido generado bajo la supervisión directa de una persona identificada,
> y bajo el marco de cumplimiento AI.Backbone Orchestrator.

## 1. Backbone TypeScript

- **Ubicación:** `src/`
- **Responsabilidad observable:** capa extensa de comandos, consultas, herramientas, tareas, servicios, pantallas y extensiones.
- **Punto de entrada observado:** no confirmado en los archivos leídos; el árbol de contexto menciona `src/main.tsx`, pero ese archivo no está presente en la ruta física consultada.
- **Capas:** aparecen agrupaciones como `commands`, `components`, `services`, `state`, `tools`, `screens` y `server`.
- **Estado:** estructura relevante, pero sin manifiesto Node visible en la raíz y sin dependencias verificables desde `package.json`.

## 2. Automatización SopraGP4U

- **Ubicación:** `sopra_clockin/`
- **Responsabilidad:** automatizar clock-in antes de las 10:00 y clock-out desde las 17:00 mediante Selenium.
- **Entrada principal:** `sopra_clockin/src/sopra_clockin.py`, función `main()`.
- **Flujo:** determina la acción por hora, crea Chrome o Edge WebDriver, navega al portal, detecta autenticación, intenta login, localiza el menú y pulsa el botón correspondiente.
- **Configuración:** `sopra_clockin/config/config.py`; permite selectores JSON, navegador, modo headless, dry-run, timeouts y reintentos.
- **Operación:** scripts `.bat`, `.ps1`, `deploy.py`, `quick_setup.py` y compatibilidad con Windows Task Scheduler.
- **Pruebas:** `test_sopra_clockin.py` cubre conversión de selectores, decisión temporal, creación de drivers, navegación simulada y clicks con mocks.

## 3. Scaffold de modelo Python

- **Ubicación:** `claude-code/python-model/`
- **Responsabilidad:** descargar modelos Hugging Face, entrenar o ajustar modelos causales y ejecutar chat local/offline.
- **Entradas:** `train.py`, `finetune.py`, `fetch_model.py`, `chat.py`, `offline_chat.py` y `run_local.py`.
- **Datos:** `data/` contiene datos de entrenamiento y conversaciones JSONL según la documentación.
- **Dependencias:** PyTorch, Transformers, Datasets, Accelerate, PEFT, bitsandbytes, safetensors y Hugging Face Hub.
- **Estado:** scaffold de experimentación; requiere modelo local o acceso a un repositorio compatible y recursos de hardware adecuados.

## 4. Prototipo de audio, LLM y memoria vectorial

- **Ubicación:** `src/`
- **Responsabilidad prevista:** captura de audio, reconocimiento, consulta a Ollama, generación de audio y persistencia de embeddings en ChromaDB.
- **Evidencia:** `Task.txt` enumera las clases y el flujo previsto; `src/context_db/`, `src/llm/` y `src/models/` contienen módulos experimentales.
- **Entrada:** `src/main.py` existe pero está vacío.
- **Estado:** diseño/prototipo; no se identificó una entrada funcional ni un manifiesto de dependencias propio.

## 5. Formación y ejemplos ML

- **Ubicación:** `Formation/`
- **Responsabilidad:** notebooks, enlaces, manuales y ejemplos de regresión, SVM, random forest y redes neuronales.
- **Entradas observadas:** scripts bajo `Formation/Code/`; `pipeline.py` está vacío.
- **Estado:** material formativo y experimental, no servicio desplegable.